// Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
// All Rights reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include <ceed-impl.h>
#include <string.h>
#include "ceed-cuda.cuh"

static __global__ void interp(const CeedInt nelem, const CeedInt dim, const CeedInt ncomp, const CeedInt elemsize, const CeedInt P, const CeedInt Q, const CeedInt nqpt, const CeedInt bufLen,
    const CeedTransposeMode tmode, const CeedScalar *interp1d, const CeedScalar *u, CeedScalar *v) {
  const CeedInt i = threadIdx.x;

  extern __shared__ CeedScalar s_mem[];
  CeedScalar *s_interp1d = s_mem;
  CeedScalar *s_buf1 = s_mem + P * Q;
  CeedScalar *s_buf2 = s_buf1 + bufLen;
  for (CeedInt k = i; k < P * Q; k += blockDim.x) {
    s_interp1d[k] = interp1d[k];
  }

  const CeedInt stride0 = tmode == CEED_NOTRANSPOSE ? P : 1;
  const CeedInt stride1 = tmode == CEED_NOTRANSPOSE ? 1 : Q;
  const CeedInt u_stride = ncomp * (tmode == CEED_NOTRANSPOSE ? elemsize : nqpt);
  const CeedInt v_stride = ncomp * (tmode == CEED_NOTRANSPOSE ? nqpt : elemsize);

  for (CeedInt elem = blockIdx.x; elem < nelem; elem += gridDim.x) {
    const CeedScalar *cur_u = u + elem * u_stride;
    CeedScalar *cur_v = v + elem * v_stride;
    for (CeedInt k = i; k < u_stride; k += blockDim.x) {
      s_buf1[k] = cur_u[k];
    }

    CeedInt pre = ncomp * CeedPowInt(P, dim - 1);
    CeedInt post = 1;
    for (CeedInt d = 0; d < dim; d++) {
      __syncthreads();

      const CeedScalar *in = d % 2 ? s_buf2 : s_buf1;
      CeedScalar *out = d == dim - 1 ? cur_v : (d % 2 ? s_buf1 : s_buf2);

      const CeedInt writeLen = pre * post * Q;
      for (CeedInt k = i; k < writeLen; k += blockDim.x) {
        const CeedInt c = k % post;
        const CeedInt j = (k / post) % Q;
        const CeedInt a = k / (post * Q);
        CeedScalar vk = 0;
        for (CeedInt b = 0; b < P; b++) {
          vk += s_interp1d[j * stride0 + b * stride1] * in[(a * P + b) * post + c];
        }

        out[k] = vk;
      }

      pre /= P;
      post *= Q;
    }
  }
}

static __global__ void grad(const CeedInt nelem, const CeedInt dim, const CeedInt ncomp, const CeedInt elemsize, const CeedInt P, const CeedInt Q, const CeedInt nqpt, const CeedInt bufLen,
    const CeedTransposeMode tmode, const CeedScalar *interp1d, const CeedScalar *grad1d, const CeedScalar *u, CeedScalar *v) {
  const CeedInt i = threadIdx.x;

  extern __shared__ CeedScalar s_mem[];
  CeedScalar *s_interp1d = s_mem;
  CeedScalar *s_grad1d = s_interp1d + P * Q;
  CeedScalar *s_buf1 = s_grad1d + P * Q;
  CeedScalar *s_buf2 = s_buf1 + bufLen;
  for (CeedInt k = i; k < P * Q; k += blockDim.x) {
    s_interp1d[k] = interp1d[k];
    s_grad1d[k] = grad1d[k];
  }

  const CeedInt op_stride0 = tmode == CEED_NOTRANSPOSE ? P : 1;
  const CeedInt op_stride1 = tmode == CEED_NOTRANSPOSE ? 1 : Q;
  const CeedInt u_stride = ncomp * (tmode == CEED_NOTRANSPOSE ? elemsize : nqpt * dim);
  const CeedInt v_stride = ncomp * (tmode == CEED_NOTRANSPOSE ? nqpt * dim : elemsize);

  for (CeedInt elem = blockIdx.x; elem < nelem; elem += gridDim.x) {
    const CeedScalar *cur_u = u + elem * u_stride;
    CeedScalar *cur_v = v + elem * v_stride;

    for (CeedInt dim1 = 0; dim1 < dim; dim1++) {
      CeedInt pre = ncomp * CeedPowInt(P, dim - 1);
      CeedInt post = 1;
      for (CeedInt dim2 = 0; dim2 < dim; dim2++) {
        __syncthreads();

        const CeedScalar *op = dim1 == dim2 ? s_grad1d : s_interp1d;
        const CeedScalar *in = dim2 == 0 ? cur_u : (dim2 % 2 ? s_buf2 : s_buf1);
        CeedScalar *out = dim2 == dim - 1 ? cur_v : (dim2 % 2 ? s_buf1 : s_buf2);

        const CeedInt writeLen = pre * post * Q;
        for (CeedInt k = i; k < writeLen; k += blockDim.x) {
          const CeedInt c = k % post;
          const CeedInt j = (k / post) % Q;
          const CeedInt a = k / (post * Q);
          CeedScalar vk = 0;
          for (CeedInt b = 0; b < P; b++) {
            vk += op[j * op_stride0 + b * op_stride1] * in[(a * P + b) * post + c];
          }

          if (tmode == CEED_TRANSPOSE && dim2 == dim - 1)
            out[k] += vk;
          else
            out[k] = vk;
        }

        pre /= P;
        post *= Q;
      }
      if (tmode == CEED_TRANSPOSE) {
        cur_u += nqpt * ncomp;
      } else {
        cur_v += nqpt * ncomp;
      }
    }
  }
}

static __global__ void weight(const CeedInt dim, const CeedInt Q, const CeedScalar *qweight1d, CeedScalar *v) {
  for (CeedInt d=0; d<dim; d++) {
    CeedInt pre = CeedPowInt(Q, dim-d-1), post = CeedPowInt(Q, d);
    for (CeedInt i=0; i<pre; i++) {
      for (CeedInt j=0; j<Q; j++) {
        for (CeedInt k=0; k<post; k++) {
          v[(i*Q + j)*post + k] = qweight1d[j] * (d == 0 ? 1 : v[(i*Q + j)*post + k]);
        }
      }
    }
  }
}

int CeedBasisApplyElems_Cuda(CeedBasis basis, const CeedInt nelem, const CeedInt elemsize, CeedTransposeMode tmode,
    CeedEvalMode emode, const CeedVector u, CeedVector v) {
  int ierr;
  const Ceed_Cuda* ceed = (Ceed_Cuda*)basis->ceed->data;
  CeedBasis_Cuda *data = (CeedBasis_Cuda*)basis->data;
  const CeedInt dim = basis->dim;
  const CeedInt ncomp = basis->ncomp;
  const CeedInt nqpt = CeedPowInt(basis->Q1d, dim);
  const CeedInt transpose = tmode == CEED_TRANSPOSE;
  const CeedInt P = transpose?basis->Q1d:basis->P1d;
  const CeedInt Q = transpose?basis->P1d:basis->Q1d;
  const CeedInt bufLen = ncomp * CeedPowInt(std::max(P, Q), dim);

  if (!data->ready) {
    data->ready = true;
    const CeedInt qBytes = basis->Q1d * sizeof(CeedScalar);
    ierr = cudaMalloc(&data->d_qweight1d, qBytes); CeedChk(ierr);
    ierr = cudaMemcpy(data->d_qweight1d, basis->qweight1d, qBytes, cudaMemcpyHostToDevice); CeedChk(ierr);

    const CeedInt iBytes = qBytes * basis->P1d;
    ierr = cudaMalloc(&data->d_interp1d, iBytes); CeedChk(ierr);
    ierr = cudaMemcpy(data->d_interp1d, basis->interp1d, iBytes, cudaMemcpyHostToDevice); CeedChk(ierr);

    ierr = cudaMalloc(&data->d_grad1d, iBytes); CeedChk(ierr);
    ierr = cudaMemcpy(data->d_grad1d, basis->grad1d, iBytes, cudaMemcpyHostToDevice); CeedChk(ierr);
  }

  const CeedScalar *d_u;
  CeedScalar *d_v;
  CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u);
  CeedVectorGetArray(v, CEED_MEM_DEVICE, &d_v);

  if (tmode == CEED_TRANSPOSE) {
    ierr = cudaMemset(d_v, 0, v->length * sizeof(CeedScalar)); CeedChk(ierr);
  }
  if (emode & CEED_EVAL_INTERP) {
    interp<<<nelem, ceed->optBlockSize, (P * Q + 2 * bufLen) * sizeof(CeedScalar)>>>(
        nelem, dim, ncomp, elemsize, P, Q, nqpt, bufLen, tmode,
        data->d_interp1d, d_u, d_v);
    ierr = cudaGetLastError(); CeedChk(ierr);

    if (transpose) {
      d_u += nelem * nqpt * ncomp;
    } else {
      d_v += nelem * nqpt * ncomp;
    }
  }

  if (emode & CEED_EVAL_GRAD) {
    grad<<<nelem, ceed->optBlockSize, 2 * (P * Q + bufLen) * sizeof(CeedScalar)>>>(
        nelem, dim, ncomp, elemsize, P, Q, nqpt, bufLen, tmode,
        data->d_interp1d, data->d_grad1d, d_u, d_v);
    ierr = cudaGetLastError(); CeedChk(ierr);

    if (transpose) {
      d_u += nelem * nqpt * ncomp * dim;
    } else {
      d_v += nelem * nqpt * ncomp * dim;
    }
  }

  if (emode & CEED_EVAL_WEIGHT) {
    if (tmode == CEED_TRANSPOSE)
      return CeedError(basis->ceed, 1,
          "CEED_EVAL_WEIGHT incompatible with CEED_TRANSPOSE");
    weight<<<1,1>>>(dim, basis->Q1d, data->d_qweight1d, d_v);
    ierr = cudaGetLastError(); CeedChk(ierr);
  }

  return 0;
}

static int CeedBasisApply_Cuda(CeedBasis basis, CeedTransposeMode tmode,
    CeedEvalMode emode,
    const CeedScalar *u, CeedScalar *v) {
  int ierr;

  CeedVector tmp_u, tmp_v;
  const CeedInt ncomp = basis->ncomp;
  const CeedInt dim = basis->dim;
  const CeedInt nqpt = CeedPowInt(basis->Q1d, dim);
  const CeedInt nppt = CeedPowInt(basis->P1d, dim);
  const CeedInt esize = (emode & CEED_EVAL_INTERP ? nqpt * ncomp : 0) + (emode & CEED_EVAL_GRAD ? nqpt * ncomp * dim : 0) + (emode & CEED_EVAL_WEIGHT ? nqpt : 0);
  const CeedInt usize = tmode == CEED_TRANSPOSE ? esize : nppt*ncomp;
  const CeedInt vsize = tmode == CEED_TRANSPOSE ? nppt*ncomp : esize;
  ierr = CeedVectorCreate(basis->ceed, usize, &tmp_u); CeedChk(ierr);
  ierr = CeedVectorCreate(basis->ceed, vsize, &tmp_v); CeedChk(ierr);

  if (!(emode & CEED_EVAL_WEIGHT)) {
    ierr = CeedVectorSetArray(tmp_u, CEED_MEM_HOST, CEED_USE_POINTER, (CeedScalar *)u); CeedChk(ierr);
  }
  ierr = CeedVectorSetArray(tmp_v, CEED_MEM_HOST, CEED_USE_POINTER, v); CeedChk(ierr);

  CeedBasisApplyElems_Cuda(basis, 1, usize, tmode, emode, tmp_u, tmp_v);

  ierr = CeedVectorGetArray(tmp_v, CEED_MEM_HOST, &v); CeedChk(ierr);

  ierr = CeedVectorDestroy(&tmp_u); CeedChk(ierr);
  ierr = CeedVectorDestroy(&tmp_v); CeedChk(ierr);

  return 0;
}

static int CeedBasisDestroy_Cuda(CeedBasis basis) {
  int ierr;

  CeedBasis_Cuda *data = (CeedBasis_Cuda *) basis->data;

  if (data->ready) {
    ierr = cudaFree(data->d_qweight1d); CeedChk(ierr);
    ierr = cudaFree(data->d_interp1d); CeedChk(ierr);
    ierr = cudaFree(data->d_grad1d); CeedChk(ierr);
  }

  ierr = CeedFree(&data); CeedChk(ierr);

  return 0;
}

int CeedBasisCreateTensorH1_Cuda(Ceed ceed, CeedInt dim, CeedInt P1d,
    CeedInt Q1d, const CeedScalar *interp1d,
    const CeedScalar *grad1d,
    const CeedScalar *qref1d,
    const CeedScalar *qweight1d,
    CeedBasis basis) {
  int ierr;
  CeedBasis_Cuda *data;
  ierr = CeedCalloc(1,&data); CeedChk(ierr);
  basis->data = data;
  data->ready = false;

  basis->Apply = CeedBasisApply_Cuda;
  basis->Destroy = CeedBasisDestroy_Cuda;
  return 0;
}