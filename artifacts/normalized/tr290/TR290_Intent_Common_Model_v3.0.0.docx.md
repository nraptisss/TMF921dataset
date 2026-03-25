# Notice

Copyright © TM Forum 2023. All Rights Reserved.

This document and translations of it may be copied and furnished to others, and derivative works that comment on or otherwise explain it or assist in its implementation may be prepared, copied, published, and distributed, in whole or in part, without restriction of any kind, provided that the above copyright notice and this section are included on all such copies and derivative works. However, this document itself may not be modified in any way, including by removing the copyright notice or references to TM FORUM, except as needed for the purpose of developing any document or deliverable produced by a TM FORUM Collaboration Project Team (in which case the rules applicable to copyrights, as set forth in the TM FORUM IPR Policy, must be followed) or as required to translate it into languages other than English.

The limited permissions granted above are perpetual and will not be revoked by TM FORUM or its successors or assigns.

This document and the information contained herein is provided on an “AS IS” basis and TM FORUM DISCLAIMS ALL WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY WARRANTY THAT THE USE OF THE INFORMATION HEREIN WILL NOT INFRINGE ANY OWNERSHIP RIGHTS OR ANY IMPLIED WARRANTIES OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.

Direct inquiries to the TM Forum office:

181 New Road, Suite 304

Parsippany, NJ 07054, USA

Tel No. +1 862 227 1648

TM Forum Web Page: www.tmforum.org

# Executive Summary

This document describes the TR290 series of specifications covering the intent common model.

# Introduction

The intent common model contains the mandatory vocabulary of intent specification. This means that every intent manager must support at least one version of the intent common model.

## Scope

This document introduces the TR290 series of specification documents about focusing on the intent and intent report structure and specification vocabulary of the intent common model.

## Revision overview

This chapter provides a brief overview of the most significant conceptual changes between model versions to illustrate the evolution of intent modeling. An iteration of the major model number indicates revisions with substantial conceptual changes. An iteration of intermediate model number indicates minor additions, while the overall concept of the model are kept. The minor model number iterates if error corrections are needed.

### Model version 3

The third major version of the intent common model introduces functions for expressing operators. Functions provide a more intuitive way to express requirements. They also improve precision in the logical expression.

Condition objects are introduced and replace parameters. Conditions are more generic and re-useable in other cases where logical evaluations are needed, for example in context.

### Model version 2

The second major version of the intent common model introduces the concept of providing requirements through logical conditions. If and when a condition is evaluated as true, the system is compliant to the requirement. Logical operators allow concatenating conditions for defining the overall logic of overall compliance to an intent.

### Model version 1

The version one of the intent common model establishes the basic structure of intent and intent reports. Requirements are provided by means of expectations, which specify goals to be fulfilled. This version also introduces the concepts of context and information as part of the intent specification.

# Document Overview

The specification of this revision of the intent common model consists of the following documents:

# Administrative Appendix

## Document History

### Version History

### Release History

## Acknowledgments

### Guide Lead & Author

### Main Contributors

### Additional Inputs