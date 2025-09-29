/*!
 * \file moe_distribute_base.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_BASE_H
#define MOE_DISTRIBUTE_BASE_H

/* system tick: 50MHz */
#define CAL_US(tick) (((tick) * 2) / 100)

/* performance macro */
// #define USE_256_TO_1__
#ifdef USE_256_TO_1__
#pragma message("use 256 to 1")
#else
#define USE_FOR_OPT__
#define DISPATCH_USE_WRITE_SHUFFLE__
#define USE_TOKEN_COUNT_SPLIT__
#define USE_ONE_CORE_WAIT__

#ifdef USE_ONE_CORE_WAIT__
#pragma message("use one core wait")

//  #define USE_ONE_CORE_GETCUMSUM__
#endif
#ifdef USE_FOR_OPT__
#pragma message("use for optimization")
#define FOR_OPT_MAX_BS__ 64
#define FOR_OPT_MAX_MOE_RANK__ 256
#endif
// #define COMBINE_USE_DYNAMIC_QUANT
#define OPT_RANK_OFFSET 512
#define USE_WRITE_SHUFFLE
#endif

/*cycle prof*/
#define USE_CYCLE_PROF

#ifdef USE_CYCLE_PROF
#pragma message("Use cycle prof")
#define CYCLE_PROF_HEADER_LEN (50 * 128)    //
#define CYCLE_PROF_ONE_FRAME_COUNT (16)     // 每次最多支持256次打点
#define CYCLE_PROF_MAX_FRAME       (1024)    // 最多纪录1024次
#define CYCLE_PROF_HEADRE_COUNTER_OFFSET 4 // 头4个8字节留给其他作用
#define DATA_FULSH(_gm_tensor, _type) \
    AscendC::Barrier(); \
    DataCacheCleanAndInvalid<_type, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(_gm_tensor);\
    __asm__("NOP"); \
    dsb(DSB_ALL);

#define CYCLE_PROF_CLASS_DEFINE() \
    bool enableProf_{false}; \
    GlobalTensor<uint64_t> profHeader_; \
    GlobalTensor<int64_t> profDataTensor_; \
    GM_ADDR profData_{nullptr}; \
    int64_t profTime_[CYCLE_PROF_ONE_FRAME_COUNT]{}; \
    uint64_t profCounter_{0};

#define CYCLE_PROF_INIT(__header) \
    if (__header != nullptr) { \
        enableProf_ = true; \
        __gm__ uint8_t *header = ((__gm__ uint8_t *)__header); \
        profHeader_.SetGlobalBuffer((__gm__ uint64_t *)(header + (2 + GetBlockIdx()) * 128)); \
        DATA_FULSH(profHeader_, uint64_t); \
        profCounter_ = profHeader_.GetValue(0); \
        DATA_FULSH(profHeader_, uint64_t); \
        profData_ = header + CYCLE_PROF_HEADER_LEN \
            + (profCounter_ % CYCLE_PROF_MAX_FRAME) * (CYCLE_PROF_ONE_FRAME_COUNT * 8 * 48) \
            + GetBlockIdx() * (CYCLE_PROF_ONE_FRAME_COUNT * 8); \
        profDataTensor_.SetGlobalBuffer((__gm__ int64_t *)profData_); \
    }

#define CYCLE_PROF_RECORD(_id) \
    if (enableProf_ && _id < CYCLE_PROF_ONE_FRAME_COUNT) { \
        pipe_barrier(PIPE_ALL); \
        auto cycle = GetSystemCycle(); \
        profTime_[_id] = cycle; \
    }

#define CYCL_PROF_INC(_c) \
    if (enableProf_) { \
        profTime_[15] += (uint64_t)_c; \
    }

#define CYCLE_PROF_FINI() \
    if (enableProf_) { \
        tpipe_->Reset(); \
        TBuf<> profBuf_; \
        tpipe_->InitBuffer(profBuf_, CYCLE_PROF_ONE_FRAME_COUNT * 16); \
        /* copy prof */ \
        auto dataLocalTensor = profBuf_.GetWithOffset<int64_t>(CYCLE_PROF_ONE_FRAME_COUNT * 8, 0); \
        int64_t preCycle = profTime_[0]; \
        for (int i = 0; i < 8; ++i) { \
            int64_t cycle = profTime_[i]; \
            dataLocalTensor.SetValue(i, (cycle - preCycle) / 50); \
        } \
        pipe_barrier(PIPE_ALL); \
        DataCopy(profDataTensor_, dataLocalTensor, CYCLE_PROF_ONE_FRAME_COUNT); \
        pipe_barrier(PIPE_ALL); \
        SyncAll<true>(); \
        /* copy counter */ \
        auto counterLocalTensor = profBuf_.GetWithOffset<uint64_t>(CYCLE_PROF_ONE_FRAME_COUNT * 8, CYCLE_PROF_ONE_FRAME_COUNT * 8); \
        counterLocalTensor.SetValue(0, profCounter_ + 1); \
        pipe_barrier(PIPE_ALL); \
        DataCopy(profHeader_, counterLocalTensor, 16); \
        pipe_barrier(PIPE_ALL); \
        printf("[RANK %d AIC %d] Init %u InputToShare %u SetStatus %u WaitStatus1 %u WaitStatus2 %u WaitStatus3 %u ShareToOutput %u\n", epRankId, blockIdx, dataLocalTensor.GetValue(1), dataLocalTensor.GetValue(2), dataLocalTensor.GetValue(3), dataLocalTensor.GetValue(4), dataLocalTensor.GetValue(5), dataLocalTensor.GetValue(6), dataLocalTensor.GetValue(7)); \
    }
    // AscendC::DumpTensor(dataLocalTensor, epRankId, CYCLE_PROF_ONE_FRAME_COUNT);
#else
#pragma message("orignal version")
#define CYCLE_PROF_CLASS_DEFINE()
#define CYCLE_PROF_INIT(__head)
#define CYCLE_PROF_RECORD(_facker_id)
#define CYCLE_PROF_FINI()
#define CYCL_PROF_INC(_c)
#endif

constexpr uint32_t LOCAL_NOTIFY_MAX_NUM = 64;
constexpr uint32_t LOCAL_STREAM_MAX_NUM = 19;
constexpr uint32_t AICPU_OP_NOTIFY_MAX_NUM = 2;
constexpr uint32_t AICPU_MAX_RANK_NUM = 128 * 1024;

struct HcclSignalInfo {
    uint64_t resId;
    uint64_t addr;
    uint32_t devId;
    uint32_t tsId;
    uint32_t rankId;
    uint32_t flag;
};

struct ListCommon {
    uint64_t nextHost;
    uint64_t preHost;
    uint64_t nextDevice;
    uint64_t preDevice;
};

struct HcclStreamInfo {
    int32_t streamIds;
    uint32_t sqIds;
    uint32_t cqIds;
    uint32_t logicCqids;
};

struct LocalResInfoV2 {
    uint32_t streamNum;
    uint32_t signalNum;
    HcclSignalInfo localSignals[LOCAL_NOTIFY_MAX_NUM];
    HcclStreamInfo streamInfo[LOCAL_STREAM_MAX_NUM];
    HcclStreamInfo mainStreamInfo;
    HcclSignalInfo aicpuOpNotify[AICPU_OP_NOTIFY_MAX_NUM];
    ListCommon nextTagRes;  // HccltagLocalResV2
};

enum class rtFloatOverflowMode_t {
    RT_OVERFLOW_MODE_SATURATION = 0,
    RT_OVERFLOW_MODE_INFNAN,
    RT_OVERFLOW_MODE_UNDEF,
};

struct AlgoTopoInfo {
    uint32_t userRank;      // RankID
    uint32_t userRankSize;  // Rank Number
    int32_t deviceLogicId;
    bool isSingleMeshAggregation;
    uint32_t deviceNumPerAggregation;
    uint32_t superPodNum;
    uint32_t devicePhyId;
    uint32_t topoType;  // TopoType
    uint32_t deviceType;
    uint32_t serverNum;
    uint32_t meshAggregationRankSize;
    uint32_t multiModuleDiffDeviceNumMode;
    uint32_t multiSuperPodDiffServerNumMode;
    uint32_t realUserRank;
    bool isDiffDeviceModule;
    bool isDiffDeviceType;
    uint32_t gcdDeviceNumPerAggregation;
    uint32_t moduleNum;
    uint32_t isUsedRdmaRankPairNum;
    uint64_t isUsedRdmaRankPair;
    uint32_t pairLinkCounterNum;
    uint64_t pairLinkCounter;
    uint32_t nicNum;
    uint64_t nicList;
    uint64_t complanRankLength;
    uint64_t complanRank;
    uint64_t bridgeRankNum;
    uint64_t bridgeRank;
    uint64_t serverAndsuperPodRankLength;
    uint64_t serverAndsuperPodRank;
};

struct HcclOpConfig {
    uint8_t deterministic;
    uint8_t retryEnable;
    uint8_t highPerfEnable;
    uint8_t padding[5];
    uint8_t linkTimeOut[8];
    uint64_t notifyWaitTime;
    uint32_t retryHoldTime;
    uint32_t retryIntervalTime;
    bool interHccsDisable = false;
    rtFloatOverflowMode_t floatOverflowMode = rtFloatOverflowMode_t::RT_OVERFLOW_MODE_UNDEF;
    uint32_t multiQpThreshold = 512;
};

struct HcclMC2WorkSpace {
    uint64_t workSpace;
    uint64_t workSpaceSize;
};

struct RemoteResPtr {
    uint64_t nextHostPtr;
    uint64_t nextDevicePtr;
};

struct HDCommunicateParams {
    uint64_t hostAddr{0};
    uint64_t deviceAddr{0};
    uint64_t readCacheAddr{0};
    uint32_t devMemSize{0};
    uint32_t buffLen{0};
    uint32_t flag{0};
};

struct HcclRankRelationResV2 {
    uint32_t remoteUsrRankId;
    uint32_t remoteWorldRank;
    uint64_t windowsIn;
    uint64_t windowsOut;
    uint64_t windowsExp;
    ListCommon nextTagRes;
};

struct HcclOpResParam {
    // local resource
    HcclMC2WorkSpace mc2WorkSpace;
    uint32_t localUsrRankId;  // usrrankid
    uint32_t rankSize;
    uint64_t winSize;
    uint64_t localWindowsIn;
    uint64_t localWindowsOut;
    char hcomId[128];
    // aicore detect remote window
    uint64_t winExpSize;
    uint64_t localWindowsExp;
    uint32_t rWinStart;
    uint32_t rWinOffset;
    uint64_t version;
    LocalResInfoV2 localRes;
    AlgoTopoInfo topoInfo;

    // config parameters
    HcclOpConfig config;
    uint64_t hostStateInfo;
    uint64_t aicpuStateInfo;
    uint64_t lockAddr;
    uint32_t rsv[16];
    uint32_t notifysize;
    uint32_t remoteResNum;
    RemoteResPtr remoteRes[AICPU_MAX_RANK_NUM];

    // communicate retry
    HDCommunicateParams kfcControlTransferH2DParams;
    HDCommunicateParams kfcStatusTransferD2HParams;
    uint64_t tinyMem;  // for all2all
    uint64_t tinyMemSize;
    // zero-copy
    uint64_t zeroCopyHeadPtr;
    uint64_t zeroCopyTailPtr;
    uint64_t zeroCopyRingBuffer;
    uint64_t zeroCopyIpcPtrs[16];
    uint32_t zeroCopyDevicePhyId[16];

    bool utraceStatusFlag;
};

#endif  // MOE_DISTRIBUTE_BASE_H
