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

#include <string.h>
#include "ceed-blocked.h"

static int CeedInit_Blocked(const char *resource, Ceed ceed) {
  int ierr;
  if (strcmp(resource, "/cpu/self")
      && strcmp(resource, "/cpu/self/blocked"))
    return CeedError(ceed, 1, "Blocked backend cannot use resource: %s", resource);

  Ceed ceedref;

  // Create refrence CEED that implementation will be dispatched
  //   through unless overridden
  CeedInit("/cpu/self/ref", &ceedref);
  ierr = CeedSetDelegate(ceed, &ceedref); CeedChk(ierr);

  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateTensorH1",
                                CeedBasisCreateTensorH1_Blocked); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateH1",
                                CeedBasisCreateH1_Blocked); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "OperatorCreate",
                                CeedOperatorCreate_Blocked); CeedChk(ierr);

  return 0;
}

__attribute__((constructor))
static void Register(void) {
  CeedRegister("/cpu/self/blocked", CeedInit_Blocked, 10);
}
