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
#include "ceed-opt.h"
#include "../ref/ceed-ref.h"

static int CeedOperatorDestroy_Opt(CeedOperator op) {
  CeedOperator_Opt *impl = op->data;
  int ierr;

  for (CeedInt i=0; i<impl->numein+impl->numeout; i++) {
    ierr = CeedElemRestrictionDestroy(&impl->blkrestr[i]); CeedChk(ierr);
    ierr = CeedVectorDestroy(&impl->evecs[i]); CeedChk(ierr);
  }
  ierr = CeedFree(&impl->evecs); CeedChk(ierr);
  ierr = CeedFree(&impl->edata); CeedChk(ierr);

  for (CeedInt i=0; i<impl->numqin+impl->numqout; i++) {
    ierr = CeedFree(&impl->qdata_alloc[i]); CeedChk(ierr);
  }
  ierr = CeedFree(&impl->qdata_alloc); CeedChk(ierr);
  ierr = CeedFree(&impl->qdata); CeedChk(ierr);

  ierr = CeedFree(&impl->indata); CeedChk(ierr);
  ierr = CeedFree(&impl->outdata); CeedChk(ierr);

  ierr = CeedFree(&op->data); CeedChk(ierr);
  return 0;
}

/*
  Setup infields or outfields
 */
static int CeedOperatorSetupFields_Opt(struct CeedQFunctionField qfields[16],
                                       struct CeedOperatorField ofields[16],
                                       CeedElemRestriction *blkrestr,
                                       CeedVector *evecs, CeedScalar **qdata,
                                       CeedScalar **qdata_alloc, CeedScalar **indata,
                                       CeedInt starti, CeedInt starte,
                                       CeedInt startq, CeedInt numfields, CeedInt Q) {
  CeedInt dim, ierr, ie=starte, iq=startq, ncomp;
  const CeedInt blksize = 8;

  // Loop over fields
  for (CeedInt i=0; i<numfields; i++) {
    if (ofields[i].Erestrict != CEED_RESTRICTION_IDENTITY) {
      CeedElemRestriction r = ofields[i].Erestrict;
      CeedElemRestriction_Ref *data = r->data;
      ierr = CeedElemRestrictionCreateBlocked(r->ceed, r->nelem, r->elemsize,
                                              blksize, r->ndof, r->ncomp,
                                              CEED_MEM_HOST, CEED_COPY_VALUES,
                                              data->indices, &blkrestr[ie]);
      ierr = CeedElemRestrictionCreateVector(blkrestr[ie], NULL, &evecs[ie]);
      CeedChk(ierr);
      ie++;
    }
    CeedEvalMode emode = qfields[i].emode;
    switch(emode) {
    case CEED_EVAL_NONE:
      break; // No action
    case CEED_EVAL_INTERP:
      ncomp = qfields[i].ncomp;
      ierr = CeedMalloc(Q*ncomp*blksize, &qdata_alloc[iq]); CeedChk(ierr);
      qdata[i + starti] = qdata_alloc[iq];
      iq++;
      break;
    case CEED_EVAL_GRAD:
      ncomp = qfields[i].ncomp;
      dim = ofields[i].basis->dim;
      ierr = CeedMalloc(Q*ncomp*dim*blksize, &qdata_alloc[iq]); CeedChk(ierr);
      qdata[i + starti] = qdata_alloc[iq];
      iq++;
      break;
    case CEED_EVAL_WEIGHT: // Only on input fields
      ierr = CeedMalloc(Q*blksize, &qdata_alloc[iq]); CeedChk(ierr);
      ierr = CeedBasisApply(ofields[iq].basis, blksize, CEED_NOTRANSPOSE,
                            CEED_EVAL_WEIGHT, NULL, qdata_alloc[iq]);
      CeedChk(ierr);
      qdata[i] = qdata_alloc[iq];
      indata[i] = qdata[i];
      iq++;
      break;
    case CEED_EVAL_DIV:
      break; // Not implimented
    case CEED_EVAL_CURL:
      break; // Not implimented
    }
  }
  return 0;
}

/*
  CeedOperator needs to connect all the named fields (be they active or passive)
  to the named inputs and outputs of its CeedQFunction.
 */
static int CeedOperatorSetup_Opt(CeedOperator op) {
  if (op->setupdone) return 0;
  CeedOperator_Opt *opopt = op->data;
  CeedQFunction qf = op->qf;
  CeedInt Q = op->numqpoints;
  int ierr;

  // Count infield and outfield array sizes and evectors
  for (CeedInt i=0; i<qf->numinputfields; i++) {
    CeedEvalMode emode = qf->inputfields[i].emode;
    opopt->numqin += !!(emode & CEED_EVAL_INTERP) + !!(emode & CEED_EVAL_GRAD) + !!
                     (emode & CEED_EVAL_WEIGHT);
    opopt->numein +=
      !!op->inputfields[i].Erestrict; // Need E-vector when restriction exists
  }
  for (CeedInt i=0; i<qf->numoutputfields; i++) {
    CeedEvalMode emode = qf->outputfields[i].emode;
    opopt->numqout += !!(emode & CEED_EVAL_INTERP) + !!(emode & CEED_EVAL_GRAD);
    opopt->numeout += !!op->outputfields[i].Erestrict;
  }

  // Allocate
  ierr = CeedCalloc(opopt->numein + opopt->numeout, &opopt->blkrestr); CeedChk(ierr);
  ierr = CeedCalloc(opopt->numein + opopt->numeout, &opopt->evecs); CeedChk(ierr);
  ierr = CeedCalloc(qf->numinputfields + qf->numoutputfields, &opopt->edata);
  CeedChk(ierr);

  ierr = CeedCalloc(opopt->numqin + opopt->numqout, &opopt->qdata_alloc);
  CeedChk(ierr);
  ierr = CeedCalloc(qf->numinputfields + qf->numoutputfields, &opopt->qdata);
  CeedChk(ierr);

  ierr = CeedCalloc(16, &opopt->indata); CeedChk(ierr);
  ierr = CeedCalloc(16, &opopt->outdata); CeedChk(ierr);

  // Set up infield and outfield pointer arrays
  // Infields
  ierr = CeedOperatorSetupFields_Opt(qf->inputfields, op->inputfields, opopt->blkrestr,
                                     opopt->evecs, opopt->qdata, opopt->qdata_alloc,
                                     opopt->indata, 0, 0, 0,
                                     qf->numinputfields, Q); CeedChk(ierr);

  // Outfields
  ierr = CeedOperatorSetupFields_Opt(qf->outputfields, op->outputfields, opopt->blkrestr,
                                     opopt->evecs, opopt->qdata, opopt->qdata_alloc,
                                     opopt->indata, qf->numinputfields, opopt->numein,
                                     opopt->numqin, qf->numoutputfields, Q); CeedChk(ierr);

  op->setupdone = 1;

  return 0;
}

static int CeedOperatorApply_Opt(CeedOperator op, CeedVector invec,
                                 CeedVector outvec, CeedRequest *request) {
  CeedOperator_Opt *opopt = op->data;
  const CeedInt blksize = 8;
  CeedInt Q = op->numqpoints, e, elemsize,
          nblks = (op->numelements / blksize) + !!(op->numelements % blksize);
  int ierr;
  CeedQFunction qf = op->qf;
  CeedTransposeMode lmode = CEED_NOTRANSPOSE;
  CeedScalar *vec_temp;

  // Setup
  ierr = CeedOperatorSetup_Opt(op); CeedChk(ierr);

  // Input Evecs and Restriction
  for (CeedInt i=0,iein=0; i<qf->numinputfields; i++) {
    // No Restriction
    if (op->inputfields[i].Erestrict == CEED_RESTRICTION_IDENTITY) {
      CeedEvalMode emode = qf->inputfields[i].emode;
      if (emode == CEED_EVAL_WEIGHT) {
      } else {
        // Active
        if (op->inputfields[i].vec == CEED_VECTOR_ACTIVE) {
          ierr = CeedVectorGetArrayRead(invec, CEED_MEM_HOST,
                                        (const CeedScalar **) &opopt->edata[i]); CeedChk(ierr);
          // Passive
        } else {
          ierr = CeedVectorGetArrayRead(op->inputfields[i].vec, CEED_MEM_HOST,
                                        (const CeedScalar **) &opopt->edata[i]); CeedChk(ierr);
        }
      }
    } else {
      // Restriction
      // Zero evec
      ierr = CeedVectorGetArray(opopt->evecs[iein], CEED_MEM_HOST, &vec_temp);
      CeedChk(ierr);
      for (CeedInt j=0; j<opopt->evecs[iein]->length; j++)
        vec_temp[j] = 0.;
      ierr = CeedVectorRestoreArray(opopt->evecs[iein], &vec_temp); CeedChk(ierr);
      // Active
      if (op->inputfields[i].vec == CEED_VECTOR_ACTIVE) {
        // Restrict
        ierr = CeedElemRestrictionApply(opopt->blkrestr[iein], CEED_NOTRANSPOSE,
                                        lmode, invec, opopt->evecs[iein],
                                        request); CeedChk(ierr);
        // Get evec
        ierr = CeedVectorGetArrayRead(opopt->evecs[iein], CEED_MEM_HOST,
                                      (const CeedScalar **) &opopt->edata[i]); CeedChk(ierr);
        iein++;
      } else {
        // Passive
        // Restrict
        ierr = CeedElemRestrictionApply(opopt->blkrestr[iein], CEED_NOTRANSPOSE,
                                        lmode, op->inputfields[i].vec, opopt->evecs[iein],
                                        request); CeedChk(ierr);
        // Get evec
        ierr = CeedVectorGetArrayRead(opopt->evecs[iein], CEED_MEM_HOST,
                                      (const CeedScalar **) &opopt->edata[i]); CeedChk(ierr);
        iein++;
      }
    }
  }

  // Output Evecs
  for (CeedInt i=0,ieout=opopt->numein; i<qf->numoutputfields; i++) {
    // No Restriction
    if (op->outputfields[i].Erestrict == CEED_RESTRICTION_IDENTITY) {
      // Active
      if (op->outputfields[i].vec == CEED_VECTOR_ACTIVE) {
        ierr = CeedVectorGetArray(outvec, CEED_MEM_HOST,
                                  &opopt->edata[i + qf->numinputfields]); CeedChk(ierr);
      } else {
        // Passive
        ierr = CeedVectorGetArray(op->outputfields[i].vec, CEED_MEM_HOST,
                                  &opopt->edata[i + qf->numinputfields]); CeedChk(ierr);
      }
    } else {
      // Restriction
      ierr = CeedVectorGetArray(opopt->evecs[ieout], CEED_MEM_HOST,
                                &opopt->edata[i + qf->numinputfields]); CeedChk(ierr);
      ieout++;
    }
  }

  // Loop through elements
  for (CeedInt b=0; b<nblks; b++) {
    e = b*blksize;
    // Input basis apply if needed
    for (CeedInt i=0; i<qf->numinputfields; i++) {
      // Get elemsize
      if (op->inputfields[i].Erestrict) {
        elemsize = op->inputfields[i].Erestrict->elemsize;
      } else {
        elemsize = Q;
      }
      // Get emode, ncomp
      CeedEvalMode emode = qf->inputfields[i].emode;
      CeedInt ncomp = qf->inputfields[i].ncomp;
      // Basis action
      switch(emode) {
      case CEED_EVAL_NONE:
        break; // No action
      case CEED_EVAL_INTERP:
        ierr = CeedBasisApply(op->inputfields[i].basis, blksize, CEED_NOTRANSPOSE,
                              CEED_EVAL_INTERP, &opopt->edata[i][e*elemsize*ncomp], opopt->qdata[i]);
        CeedChk(ierr);
        break;
      case CEED_EVAL_GRAD:
        ierr = CeedBasisApply(op->inputfields[i].basis, blksize, CEED_NOTRANSPOSE,
                              CEED_EVAL_GRAD, &opopt->edata[i][e*elemsize*ncomp], opopt->qdata[i]);
        CeedChk(ierr);
        break;
      case CEED_EVAL_WEIGHT:
        break;  // No action
      case CEED_EVAL_DIV:
        break; // Not implimented
      case CEED_EVAL_CURL:
        break; // Not implimented
      }
    }
    // Input pointers
    for (CeedInt i=0; i<qf->numinputfields; i++) {
      // Get emode, ncomp
      CeedEvalMode emode = qf->inputfields[i].emode;
      CeedInt ncomp = qf->inputfields[i].ncomp;
      // Basis action
      switch(emode) {
      case CEED_EVAL_NONE:
        opopt->indata[i] = &opopt->edata[i][e*Q*ncomp];
        break;
      case CEED_EVAL_INTERP:
        opopt->indata[i] = opopt->qdata[i];
        break;
      case CEED_EVAL_GRAD:
        opopt->indata[i] = opopt->qdata[i];
        break;
      case CEED_EVAL_WEIGHT:
        break;  // No action
      case CEED_EVAL_DIV:
        break; // Not implimented
      case CEED_EVAL_CURL:
        break; // Not implimented
      }
    }
    // Output pointers
    for (CeedInt i=0; i<qf->numoutputfields; i++) {
      // Get emode, ncomp
      CeedEvalMode emode = qf->outputfields[i].emode;
      CeedInt ncomp = qf->outputfields[i].ncomp;
      // Basis action
      switch(emode) {
      case CEED_EVAL_NONE:
        opopt->outdata[i] = &opopt->edata[i + qf->numinputfields][e*Q*ncomp];
        break;
      case CEED_EVAL_INTERP:
        opopt->outdata[i] = opopt->qdata[i + qf->numinputfields];
        break;
      case CEED_EVAL_GRAD:
        opopt->outdata[i] = opopt->qdata[i + qf->numinputfields];
        break;
      case CEED_EVAL_WEIGHT:
        break;  // Should not occur
      case CEED_EVAL_DIV:
        break; // Not implimented
      case CEED_EVAL_CURL:
        break; // Not implimented
      }
    }
    // Q function
    ierr = CeedQFunctionApply(op->qf, Q*blksize,
                              (const CeedScalar * const*) opopt->indata,
                              opopt->outdata); CeedChk(ierr);

    // Output basis apply if needed
    for (CeedInt i=0; i<qf->numoutputfields; i++) {
      // Get elemsize
      if (op->outputfields[i].Erestrict) {
        elemsize = op->outputfields[i].Erestrict->elemsize;
      } else {
        elemsize = Q;
      }
      // Get emode, ncomp
      CeedInt ncomp = qf->outputfields[i].ncomp;
      CeedEvalMode emode = qf->outputfields[i].emode;
      // Basis action
      switch(emode) {
      case CEED_EVAL_NONE:
        break; // No action
      case CEED_EVAL_INTERP:
        ierr = CeedBasisApply(op->outputfields[i].basis, blksize, CEED_TRANSPOSE,
                              CEED_EVAL_INTERP, opopt->qdata[i + qf->numinputfields],
                              &opopt->edata[i + qf->numinputfields][e*elemsize*ncomp]); CeedChk(ierr);
        break;
      case CEED_EVAL_GRAD:
        ierr = CeedBasisApply(op->outputfields[i].basis, blksize, CEED_TRANSPOSE,
                              CEED_EVAL_GRAD, opopt->qdata[i + qf->numinputfields],
                              &opopt->edata[i + qf->numinputfields][e*elemsize*ncomp]);
        CeedChk(ierr);
        break;
      case CEED_EVAL_WEIGHT:
        break; // Should not occur
      case CEED_EVAL_DIV:
        break; // Not implimented
      case CEED_EVAL_CURL:
        break; // Not implimented
      }
    }
  }

  // Output restriction
  for (CeedInt i=0,ieout=opopt->numein; i<qf->numoutputfields; i++) {
    // No Restriction
    if (op->outputfields[i].Erestrict == CEED_RESTRICTION_IDENTITY) {
      // Active
      if (op->outputfields[i].vec == CEED_VECTOR_ACTIVE) {
        ierr = CeedVectorRestoreArray(outvec, &opopt->edata[i + qf->numinputfields]);
        CeedChk(ierr);
      } else {
        // Passive
        ierr = CeedVectorRestoreArray(op->outputfields[i].vec,
                                      &opopt->edata[i + qf->numinputfields]); CeedChk(ierr);
      }
    } else {
      // Restriction
      // Active
      if (op->outputfields[i].vec == CEED_VECTOR_ACTIVE) {
        // Restore evec
        ierr = CeedVectorRestoreArray(opopt->evecs[ieout],
                                      &opopt->edata[i + qf->numinputfields]); CeedChk(ierr);
        // Zero lvec
        ierr = CeedVectorGetArray(outvec, CEED_MEM_HOST, &vec_temp); CeedChk(ierr);
        for (CeedInt j=0; j<outvec->length; j++)
          vec_temp[j] = 0.;
        ierr = CeedVectorRestoreArray(outvec, &vec_temp); CeedChk(ierr);
        // Restrict
        ierr = CeedElemRestrictionApply(opopt->blkrestr[ieout], CEED_TRANSPOSE,
                                        lmode, opopt->evecs[ieout], outvec, request); CeedChk(ierr);
        ieout++;
      } else {
        // Passive
        // Restore evec
        ierr = CeedVectorRestoreArray(opopt->evecs[ieout],
                                      &opopt->edata[i + qf->numinputfields]); CeedChk(ierr);
        // Zero lvec
        ierr = CeedVectorGetArray(op->outputfields[i].vec, CEED_MEM_HOST, &vec_temp);
        CeedChk(ierr);
        for (CeedInt j=0; j<op->outputfields[i].vec->length; j++)
          vec_temp[j] = 0.;
        ierr = CeedVectorRestoreArray(op->outputfields[i].vec, &vec_temp);
        CeedChk(ierr);
        // Restrict
        ierr = CeedElemRestrictionApply(opopt->blkrestr[ieout], CEED_TRANSPOSE,
                                        lmode, opopt->evecs[ieout], op->outputfields[i].vec,
                                        request); CeedChk(ierr);
        ieout++;
      }
    }
  }

  // Restore input arrays
  for (CeedInt i=0,iein=0; i<qf->numinputfields; i++) {
    // No Restriction
    if (op->inputfields[i].Erestrict == CEED_RESTRICTION_IDENTITY) {
      CeedEvalMode emode = qf->inputfields[i].emode;
      if (emode & CEED_EVAL_WEIGHT) {
      } else {
        // Active
        if (op->inputfields[i].vec == CEED_VECTOR_ACTIVE) {
          ierr = CeedVectorRestoreArrayRead(invec,
                                            (const CeedScalar **) &opopt->edata[i]); CeedChk(ierr);
          // Passive
        } else {
          ierr = CeedVectorRestoreArrayRead(op->inputfields[i].vec,
                                            (const CeedScalar **) &opopt->edata[i]); CeedChk(ierr);
        }
      }
    } else {
      // Restriction
      ierr = CeedVectorRestoreArrayRead(opopt->evecs[iein],
                                        (const CeedScalar **) &opopt->edata[i]); CeedChk(ierr);
      iein++;
    }
  }


  return 0;
}

int CeedOperatorCreate_Opt(CeedOperator op) {
  CeedOperator_Opt *impl;
  int ierr;

  ierr = CeedCalloc(1, &impl); CeedChk(ierr);
  op->data = impl;
  op->Destroy = CeedOperatorDestroy_Opt;
  op->Apply = CeedOperatorApply_Opt;
  return 0;
}
