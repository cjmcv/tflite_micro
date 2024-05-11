#include <stdio.h>
#include <vector>
#include <stdlib.h>

extern int ActivationsTest();
extern int AddNTest();
extern int AddTest();
extern int ArgMinMaxTest();

extern int BatchMatmulTest();
extern int BatchToSpaceNDTest();
extern int BroadcastArgsTest();
extern int BroadcastToTest();

extern int CastTest();
extern int CeilTest();
extern int ComparisonsTest();
extern int ConcatenationTest();
extern int ConvTest();
extern int CumsumTest();

extern int DepthToSpaceTest();
extern int DepthwiseConvTest();
extern int DequantizeTest();
extern int DivTest();

extern int ElementwiseTest();
extern int EluTest();
extern int ExpTest();
extern int ExpandDimsTest();

extern int FillTest();
extern int FloorDivTest();
extern int FloorModTest();
extern int FloorTest();
extern int FullyConnectedTest();

extern int GatherNdTest();
extern int GatherTest();
extern int HardSwishTest();

extern int L2Pool2dTest();
extern int L2NormTest();
extern int LeakyReluTest();
extern int LogSoftmaxTest();
extern int LogicalTest();
extern int LogisticTest();
extern int LstmEvalTest();

extern int MaximumMinimumTest();
extern int MirrorPadTest();
extern int MulTest();

extern int NegTest();
extern int PackTest();
extern int PadTest();
extern int PoolingTest();
extern int PreluTest();

extern int QuantizeTest();
extern int ReduceTest();
extern int ReshapeTest();
extern int ResizeBilinearTest();
extern int ResizeNearestNeighborTest();
extern int RoundTest();

extern int SelectTest();
extern int ShapeTest();
extern int SliceTest();
extern int SoftmaxTest();
extern int SpaceToBatchNdTest();
extern int SpaceToDepthFuncTest();
extern int SplitTest();
extern int SplitVTest();
extern int SquaredDifferenceTest();
extern int SqueezeTest();
extern int StridedSliceTest();
extern int SubTest();
extern int SvdfTest();

extern int TanhTest();
extern int TransposeConvTest();
extern int TransposeTest();
extern int UnidirectionalSequenceLstmTest();
extern int UnpackTest();
extern int ZerosLikeTest();

#define CHECK_UNITTEST_SUCCESS(x) \
    if (x) std::abort();

void TestKernels() {

    // UintTestFuncs::GetInstance()->Push(&ActivationsTest);
    CHECK_UNITTEST_SUCCESS(ActivationsTest());
    CHECK_UNITTEST_SUCCESS(AddTest());
    CHECK_UNITTEST_SUCCESS(AddNTest());
    CHECK_UNITTEST_SUCCESS(ArgMinMaxTest());

    CHECK_UNITTEST_SUCCESS(BatchMatmulTest());
    CHECK_UNITTEST_SUCCESS(BatchToSpaceNDTest());
    CHECK_UNITTEST_SUCCESS(BroadcastArgsTest());
    CHECK_UNITTEST_SUCCESS(BroadcastToTest());

    CHECK_UNITTEST_SUCCESS(CastTest());
    CHECK_UNITTEST_SUCCESS(CeilTest());
    CHECK_UNITTEST_SUCCESS(ComparisonsTest());
    CHECK_UNITTEST_SUCCESS(ConcatenationTest());
    CHECK_UNITTEST_SUCCESS(ConvTest());
    CHECK_UNITTEST_SUCCESS(CumsumTest());

    CHECK_UNITTEST_SUCCESS(DepthToSpaceTest());
    CHECK_UNITTEST_SUCCESS(DepthwiseConvTest());
    CHECK_UNITTEST_SUCCESS(DequantizeTest());
    CHECK_UNITTEST_SUCCESS(DivTest());

    CHECK_UNITTEST_SUCCESS(ElementwiseTest());
    CHECK_UNITTEST_SUCCESS(EluTest());
    CHECK_UNITTEST_SUCCESS(ExpTest());
    CHECK_UNITTEST_SUCCESS(ExpandDimsTest());

    CHECK_UNITTEST_SUCCESS(FillTest());
    CHECK_UNITTEST_SUCCESS(FloorDivTest());
    CHECK_UNITTEST_SUCCESS(FloorModTest());
    CHECK_UNITTEST_SUCCESS(FloorTest());
    CHECK_UNITTEST_SUCCESS(FullyConnectedTest());
    
    CHECK_UNITTEST_SUCCESS(L2Pool2dTest());
    CHECK_UNITTEST_SUCCESS(L2NormTest());
    CHECK_UNITTEST_SUCCESS(LeakyReluTest());
    CHECK_UNITTEST_SUCCESS(LogSoftmaxTest());
    CHECK_UNITTEST_SUCCESS(LogicalTest());
    CHECK_UNITTEST_SUCCESS(LogisticTest());
    CHECK_UNITTEST_SUCCESS(LstmEvalTest());

    CHECK_UNITTEST_SUCCESS(MaximumMinimumTest());
    CHECK_UNITTEST_SUCCESS(MirrorPadTest());
    CHECK_UNITTEST_SUCCESS(MulTest());

    CHECK_UNITTEST_SUCCESS(NegTest());
    CHECK_UNITTEST_SUCCESS(PackTest());
    CHECK_UNITTEST_SUCCESS(PadTest());
    CHECK_UNITTEST_SUCCESS(PoolingTest());
    CHECK_UNITTEST_SUCCESS(PreluTest());

    CHECK_UNITTEST_SUCCESS(QuantizeTest());
    CHECK_UNITTEST_SUCCESS(ReduceTest());
    CHECK_UNITTEST_SUCCESS(ReshapeTest());
    CHECK_UNITTEST_SUCCESS(ResizeBilinearTest());
    CHECK_UNITTEST_SUCCESS(ResizeNearestNeighborTest());
    CHECK_UNITTEST_SUCCESS(RoundTest());

    CHECK_UNITTEST_SUCCESS(SelectTest());
    CHECK_UNITTEST_SUCCESS(ShapeTest());
    CHECK_UNITTEST_SUCCESS(SliceTest());
    CHECK_UNITTEST_SUCCESS(SoftmaxTest());
    CHECK_UNITTEST_SUCCESS(SpaceToBatchNdTest());
    CHECK_UNITTEST_SUCCESS(SpaceToDepthFuncTest());
    CHECK_UNITTEST_SUCCESS(SplitTest());
    CHECK_UNITTEST_SUCCESS(SplitVTest());
    CHECK_UNITTEST_SUCCESS(SquaredDifferenceTest());
    CHECK_UNITTEST_SUCCESS(SqueezeTest());
    CHECK_UNITTEST_SUCCESS(StridedSliceTest());
    CHECK_UNITTEST_SUCCESS(SubTest());
    CHECK_UNITTEST_SUCCESS(SvdfTest());

    CHECK_UNITTEST_SUCCESS(TanhTest());
    CHECK_UNITTEST_SUCCESS(TransposeConvTest());
    CHECK_UNITTEST_SUCCESS(TransposeTest());
    CHECK_UNITTEST_SUCCESS(UnidirectionalSequenceLstmTest());
    CHECK_UNITTEST_SUCCESS(UnpackTest());
    CHECK_UNITTEST_SUCCESS(ZerosLikeTest());

    printf("Finish TestKernels.\n");
}