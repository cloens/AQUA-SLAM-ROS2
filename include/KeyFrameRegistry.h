#ifndef KEYFRAME_REGISTRY_H
#define KEYFRAME_REGISTRY_H

#include <map>
#include <mutex>
#include <unordered_set>
#include <vector>

#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>

namespace ORB_SLAM3
{

class Frame;
class KeyFrame;
class Map;

class KeyFrameDatabase
{
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive& archive, const unsigned int)
    {
        archive & mBackupKeyFrameIds;
    }

public:
    KeyFrameDatabase() = default;

    void add(KeyFrame* keyframe);
    void erase(KeyFrame* keyframe);
    void clear();
    void clearMap(Map* map);

    std::vector<KeyFrame*> DetectLoopCandidates(KeyFrame*, float);
    void DetectCandidates(KeyFrame*, float, std::vector<KeyFrame*>& loop,
                          std::vector<KeyFrame*>& merge);
    void DetectBestCandidates(KeyFrame*, std::vector<KeyFrame*>& loop,
                              std::vector<KeyFrame*>& merge, int);
    void DetectNBestCandidates(KeyFrame*, std::vector<KeyFrame*>& loop,
                               std::vector<KeyFrame*>& merge, int);
    std::vector<KeyFrame*> DetectRelocalizationCandidates(Frame*, Map* map);

    void PreSave();
    void PostLoad(const std::map<long unsigned int, KeyFrame*>& keyframesById);

private:
    std::unordered_set<KeyFrame*> mKeyFrames;
    std::vector<long unsigned int> mBackupKeyFrameIds;
    std::mutex mMutex;
};

} // namespace ORB_SLAM3

#endif
