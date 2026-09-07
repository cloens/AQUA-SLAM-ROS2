#include "KeyFrameRegistry.h"

#include <algorithm>

#include "KeyFrame.h"

namespace ORB_SLAM3
{

void KeyFrameDatabase::add(KeyFrame* keyframe)
{
    if (!keyframe)
        return;
    std::lock_guard<std::mutex> lock(mMutex);
    mKeyFrames.insert(keyframe);
}

void KeyFrameDatabase::erase(KeyFrame* keyframe)
{
    std::lock_guard<std::mutex> lock(mMutex);
    mKeyFrames.erase(keyframe);
}

void KeyFrameDatabase::clear()
{
    std::lock_guard<std::mutex> lock(mMutex);
    mKeyFrames.clear();
}

void KeyFrameDatabase::clearMap(Map* map)
{
    std::lock_guard<std::mutex> lock(mMutex);
    for (auto iterator = mKeyFrames.begin(); iterator != mKeyFrames.end();) {
        if ((*iterator)->GetMap() == map)
            iterator = mKeyFrames.erase(iterator);
        else
            ++iterator;
    }
}

std::vector<KeyFrame*> KeyFrameDatabase::DetectLoopCandidates(KeyFrame*, float)
{
    return {};
}

void KeyFrameDatabase::DetectCandidates(
    KeyFrame*, float, std::vector<KeyFrame*>& loop,
    std::vector<KeyFrame*>& merge)
{
    loop.clear();
    merge.clear();
}

void KeyFrameDatabase::DetectBestCandidates(
    KeyFrame*, std::vector<KeyFrame*>& loop,
    std::vector<KeyFrame*>& merge, int)
{
    loop.clear();
    merge.clear();
}

void KeyFrameDatabase::DetectNBestCandidates(
    KeyFrame*, std::vector<KeyFrame*>& loop,
    std::vector<KeyFrame*>& merge, int)
{
    loop.clear();
    merge.clear();
}

std::vector<KeyFrame*> KeyFrameDatabase::DetectRelocalizationCandidates(
    Frame*, Map* map)
{
    std::lock_guard<std::mutex> lock(mMutex);
    std::vector<KeyFrame*> candidates;
    candidates.reserve(mKeyFrames.size());
    for (KeyFrame* keyframe : mKeyFrames) {
        if (keyframe && !keyframe->isBad() && keyframe->GetMap() == map)
            candidates.push_back(keyframe);
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const KeyFrame* first, const KeyFrame* second) {
                  return first->mnId > second->mnId;
              });
    return candidates;
}

void KeyFrameDatabase::PreSave()
{
    std::lock_guard<std::mutex> lock(mMutex);
    mBackupKeyFrameIds.clear();
    mBackupKeyFrameIds.reserve(mKeyFrames.size());
    for (const KeyFrame* keyframe : mKeyFrames)
        mBackupKeyFrameIds.push_back(keyframe->mnId);
}

void KeyFrameDatabase::PostLoad(
    const std::map<long unsigned int, KeyFrame*>& keyframesById)
{
    std::lock_guard<std::mutex> lock(mMutex);
    mKeyFrames.clear();
    for (const auto id : mBackupKeyFrameIds) {
        const auto iterator = keyframesById.find(id);
        if (iterator != keyframesById.end())
            mKeyFrames.insert(iterator->second);
    }
    mBackupKeyFrameIds.clear();
}

} // namespace ORB_SLAM3
