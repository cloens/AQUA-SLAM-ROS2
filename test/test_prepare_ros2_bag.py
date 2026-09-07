#!/usr/bin/env python3

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_types_from_msg, get_typestore


SCRIPT = Path(__file__).parents[1] / "tools" / "prepare_ros2_bag.py"
SPEC = importlib.util.spec_from_file_location("prepare_ros2_bag", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PrepareRos2BagTest(unittest.TestCase):
    def test_preserves_rejected_dvl_quality_for_backend_decision(self):
        message = SimpleNamespace(
            velocity_valid=False,
            status=5,
            fom=0.2,
            altitude=-1.0,
            beams=[
                SimpleNamespace(valid=True),
                SimpleNamespace(valid=False),
                SimpleNamespace(valid=True),
                SimpleNamespace(valid=False),
            ],
        )

        quality = MODULE.dvl_quality_fields(message, "bottom_track")

        self.assertFalse(quality["velocity_valid"])
        self.assertEqual(quality["status"], 5)
        self.assertEqual(quality["track_mode"], 1)
        self.assertEqual(quality["valid_beam_ratio"], 0.5)
        self.assertFalse(quality["altitude_valid"])
        self.assertEqual(quality["altitude"], -1.0)
        self.assertFalse(quality["covariance_valid"])
        self.assertEqual(quality["covariance"], [0.0] * 9)

    def test_builds_normalized_dvl_without_changing_source_header(self):
        store = get_typestore(Stores.ROS2_JAZZY)
        stamp = store.types["builtin_interfaces/msg/Time"](sec=12, nanosec=45)
        header = store.types["std_msgs/msg/Header"](
            stamp=stamp, frame_id="dvl_link"
        )
        velocity = store.types["geometry_msgs/msg/Vector3"](
            x=0.4, y=-0.2, z=0.1
        )
        message = SimpleNamespace(
            header=header,
            velocity=velocity,
            velocity_valid=True,
            status=0,
            fom=0.02,
            altitude=1.25,
            beams=[SimpleNamespace(valid=True) for _ in range(4)],
        )

        normalized = MODULE.normalized_dvl_message(
            message, "bottom_track", store
        )

        self.assertEqual(normalized.header.stamp.sec, 12)
        self.assertEqual(normalized.header.stamp.nanosec, 45)
        self.assertEqual(normalized.header.frame_id, "dvl_link")
        self.assertEqual(normalized.velocity.x, 0.4)
        self.assertEqual(normalized.track_mode, 1)
        self.assertTrue(normalized.velocity_valid)
        self.assertFalse(normalized.covariance_valid)
        self.assertEqual(normalized.valid_beam_ratio, 1.0)

    def test_adapted_bag_keeps_source_rejected_dvl_and_timestamps(self):
        source_type = "waterlinked_a50_ros_driver/msg/DVL"
        beam_type = "waterlinked_a50_ros_driver/msg/DVLBeam"
        source_store = get_typestore(Stores.ROS2_JAZZY)
        source_store.register(get_types_from_msg(
            """int64 id
float64 velocity
float64 distance
float64 rssi
float64 nsd
bool valid
""",
            beam_type,
        ))
        source_store.register(get_types_from_msg(
            """std_msgs/Header header
float64 time
geometry_msgs/Vector3 velocity
float64 fom
float64 altitude
waterlinked_a50_ros_driver/DVLBeam[] beams
bool velocity_valid
int64 status
string form
""",
            source_type,
        ))

        stamp = source_store.types["builtin_interfaces/msg/Time"](
            sec=12, nanosec=45
        )
        message = source_store.types[source_type](
            header=source_store.types["std_msgs/msg/Header"](
                stamp=stamp, frame_id="dvl_link"
            ),
            time=12.000000045,
            velocity=source_store.types["geometry_msgs/msg/Vector3"](
                x=0.4, y=-0.2, z=0.1
            ),
            fom=0.02,
            altitude=-1.0,
            beams=np.asarray([
                source_store.types[beam_type](
                    id=index, velocity=0.0, distance=0.0,
                    rssi=0.0, nsd=0.0, valid=index < 2,
                )
                for index in range(4)
            ], dtype=object),
            velocity_valid=False,
            status=5,
            form="",
        )

        with TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            destination = Path(directory) / "destination"
            source_timestamp = 12_000_000_045
            with Writer(source, version=9) as writer:
                connection = writer.add_connection(
                    "/dvl/data", source_type,
                    typestore=source_store, serialization_format="cdr",
                )
                writer.write(
                    connection,
                    source_timestamp,
                    source_store.serialize_cdr(message, source_type),
                )

            MODULE.adapt_dvl_topic(source, destination, "bottom_track")

            with AnyReader([destination]) as reader:
                records = list(reader.messages())
                self.assertEqual(len(records), 1)
                connection, bag_timestamp, rawdata = records[0]
                normalized = reader.deserialize(rawdata, connection.msgtype)
                self.assertEqual(connection.topic, "/uw_slam/dvl")
                self.assertEqual(
                    connection.msgtype, MODULE.DVL_OBSERVATION_TYPE
                )
                self.assertEqual(bag_timestamp, source_timestamp)
                self.assertEqual(normalized.header.stamp.sec, 12)
                self.assertEqual(normalized.header.stamp.nanosec, 45)
                self.assertFalse(normalized.velocity_valid)
                self.assertEqual(normalized.status, 5)
                self.assertEqual(normalized.valid_beam_ratio, 0.5)


if __name__ == "__main__":
    unittest.main()
