#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_types_from_msg, get_typestore


DVL_OBSERVATION_TYPE = 'uw_slam_bridge/msg/DvlObservation'
DVL_OBSERVATION_DEFINITION = """\
uint8 TRACK_MODE_UNKNOWN=0
uint8 TRACK_MODE_BOTTOM=1
uint8 TRACK_MODE_WATER=2
std_msgs/Header header
geometry_msgs/Vector3 velocity
bool velocity_valid
bool covariance_valid
float64[9] covariance
uint8 track_mode
float64 valid_beam_ratio
bool altitude_valid
float64 altitude
float64 error_velocity
int64 status
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert a ROS1 bag to rosbag2 and adapt DVL topics for AQUA-SLAM.',
    )
    parser.add_argument('src', type=Path, help='Source ROS1 bag file')
    parser.add_argument(
        '--tmp-ros2',
        type=Path,
        help='Temporary rosbag2 path. Defaults next to the source bag.',
    )
    parser.add_argument(
        '--dst',
        type=Path,
        help='Final rosbag2 path. Defaults to <src_stem>_aqua_ros2 next to the source bag.',
    )
    parser.add_argument(
        '--dvl-track-mode',
        required=True,
        choices=('bottom_track', 'water_track'),
        help='Registered Water Linked tracking mode for this acquisition.',
    )
    return parser.parse_args()


def convert_ros1_to_ros2(src: Path, tmp_ros2: Path) -> None:
    if tmp_ros2.exists():
        shutil.rmtree(tmp_ros2)

    subprocess.run(
        [
            'rosbags-convert',
            '--src',
            str(src),
            '--dst',
            str(tmp_ros2),
        ],
        check=True,
    )


def dvl_quality_fields(dvl_msg, track_mode: str) -> dict:
    """Preserve Water Linked quality evidence for backend-owned decisions."""
    track_modes = {"bottom_track": 1, "water_track": 2}
    if track_mode not in track_modes:
        raise ValueError(f"unsupported DVL track mode: {track_mode}")
    covariance = np.zeros(9, dtype=np.float64)
    valid_beam_ratio = (
        sum(bool(beam.valid) for beam in dvl_msg.beams) / len(dvl_msg.beams)
        if dvl_msg.beams else 0.0
    )
    return {
        "velocity_valid": bool(dvl_msg.velocity_valid),
        "covariance_valid": False,
        "covariance": covariance.tolist(),
        "track_mode": track_modes[track_mode],
        "valid_beam_ratio": valid_beam_ratio,
        "altitude_valid": bool(
            np.isfinite(dvl_msg.altitude) and dvl_msg.altitude >= 0.0
        ),
        "altitude": float(dvl_msg.altitude),
        "error_velocity": float(dvl_msg.fom),
        "status": int(dvl_msg.status),
    }


def normalized_dvl_message(dvl_msg, track_mode: str, typestore):
    if DVL_OBSERVATION_TYPE not in typestore.types:
        typestore.register(get_types_from_msg(
            DVL_OBSERVATION_DEFINITION, DVL_OBSERVATION_TYPE
        ))

    quality = dvl_quality_fields(dvl_msg, track_mode)
    header_type = typestore.types['std_msgs/msg/Header']
    time_type = typestore.types['builtin_interfaces/msg/Time']
    vector_type = typestore.types['geometry_msgs/msg/Vector3']
    observation_type = typestore.types[DVL_OBSERVATION_TYPE]
    return observation_type(
        header=header_type(
            stamp=time_type(
                sec=dvl_msg.header.stamp.sec,
                nanosec=dvl_msg.header.stamp.nanosec,
            ),
            frame_id=dvl_msg.header.frame_id,
        ),
        velocity=vector_type(
            x=float(dvl_msg.velocity.x),
            y=float(dvl_msg.velocity.y),
            z=float(dvl_msg.velocity.z),
        ),
        velocity_valid=quality['velocity_valid'],
        covariance_valid=quality['covariance_valid'],
        covariance=np.asarray(quality['covariance'], dtype=np.float64),
        track_mode=quality['track_mode'],
        valid_beam_ratio=quality['valid_beam_ratio'],
        altitude_valid=quality['altitude_valid'],
        altitude=quality['altitude'],
        error_velocity=quality['error_velocity'],
        status=quality['status'],
    )


def adapt_dvl_topic(src_ros2: Path, dst_ros2: Path, track_mode: str) -> None:
    if dst_ros2.exists():
        shutil.rmtree(dst_ros2)

    jazzy_store = get_typestore(Stores.ROS2_JAZZY)
    dvl_src_topic = '/dvl/data'
    dvl_dst_topic = '/uw_slam/dvl'
    jazzy_store.register(get_types_from_msg(
        DVL_OBSERVATION_DEFINITION, DVL_OBSERVATION_TYPE
    ))

    with AnyReader([src_ros2]) as reader, Writer(dst_ros2, version=9) as writer:
        conn_map = {}
        dvl_conn = None

        for conn in reader.connections:
            if conn.topic == dvl_src_topic:
                dvl_conn = conn
                continue

            conn_map[conn.id] = writer.add_connection(
                conn.topic,
                conn.msgtype,
                typestore=reader.typestore,
                serialization_format=conn.ext.serialization_format,
                offered_qos_profiles=conn.ext.offered_qos_profiles,
            )

        normalized_dvl_conn = writer.add_connection(
            dvl_dst_topic,
            DVL_OBSERVATION_TYPE,
            typestore=jazzy_store,
            serialization_format='cdr',
        )

        for conn, timestamp, rawdata in reader.messages():
            if dvl_conn is not None and conn.id == dvl_conn.id:
                dvl_msg = reader.deserialize(rawdata, conn.msgtype)
                normalized_msg = normalized_dvl_message(
                    dvl_msg, track_mode, jazzy_store
                )
                writer.write(
                    normalized_dvl_conn,
                    timestamp,
                    jazzy_store.serialize_cdr(
                        normalized_msg, DVL_OBSERVATION_TYPE
                    ),
                )
                continue

            writer.write(conn_map[conn.id], timestamp, rawdata)


def main() -> None:
    args = parse_args()
    src = args.src.resolve()
    tmp_ros2 = (args.tmp_ros2 or src.with_name(f'{src.stem}_ros2')).resolve()
    dst_ros2 = (args.dst or src.with_name(f'{src.stem}_aqua_ros2')).resolve()

    convert_ros1_to_ros2(src, tmp_ros2)
    adapt_dvl_topic(tmp_ros2, dst_ros2, args.dvl_track_mode)

    print(f'Prepared AQUA-SLAM rosbag2 at: {dst_ros2}')
    print(f'Temporary intermediate rosbag2 at: {tmp_ros2}')


if __name__ == '__main__':
    main()
