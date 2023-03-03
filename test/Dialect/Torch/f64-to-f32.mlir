// RUN: torch-mlir-opt -split-input-file -torch-f64-to-f32 %s | FileCheck %s

// CHECK-LABEL: func.func @add(%arg0: !torch.vtensor<[4,4],f32>)
// CHECK: torch.vtensor.literal(dense<2.000000e+00> : tensor<4x4xf32>) : !torch.vtensor<[4,4],f32>
// CHECK: torch.aten.add.Tensor %arg0, %0, %int1 : !torch.vtensor<[4,4],f32>, !torch.vtensor<[4,4],f32>, !torch.int -> !torch.vtensor<[4,4],f32>
// CHECK: torch.aten.to.dtype %1, %int6, %false, %false, %none : !torch.vtensor<[4,4],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[4,4],f32>
func.func @add(%arg0: !torch.vtensor<[4,4],f32>) -> !torch.vtensor<[4,4],f32> {
  %0 = torch.vtensor.literal(dense<2.0> : tensor<4x4xf64>) : !torch.vtensor<[4,4],f64>
  %int1 = torch.constant.int 1
  %int6 = torch.constant.int 6
  %none = torch.constant.none
  %false = torch.constant.bool false
  
  %1 = torch.aten.add.Tensor %arg0, %0, %int1 : !torch.vtensor<[4,4],f32>, !torch.vtensor<[4,4],f64>, !torch.int -> !torch.vtensor<[4,4],f64>
  %2 = torch.aten.to.dtype %1, %int6, %false, %false, %none : !torch.vtensor<[4,4],f64>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[4,4],f32> 
  return %2 : !torch.vtensor<[4,4],f32>
}
