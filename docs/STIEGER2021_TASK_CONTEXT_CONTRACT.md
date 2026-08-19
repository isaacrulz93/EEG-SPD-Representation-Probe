# Stieger2021 Task-Context Contract

The primary task is exactly `TrialData.tasknumber == 3`, the 2D cursor-control task containing all four target numbers in one common context. Literal semantic order is target 1 right hand, target 2 left hand, target 3 both hands/up, target 4 rest/down.

Task 1 contains only right/left and task 2 only both-hand/rest. Pooling tasks 1–3 would make semantic identity partially synonymous with block/task context and is forbidden. After the four-class primary terminal, task 1 and task 2 may be reported as separate, binary, non-voting diagnostics only.

The fields `result`, `forcedresult`, `targethitnumber`, and `performance` are sealed provenance/outcome fields. Trial duration is retained for feedback-window retention reporting, but it cannot determine primary inclusion. Primary inclusion uses task number and `artifact == 0` only, subject to sample availability and numerical/data-contract gates.

The pre-target epoch is an explicit warning control. Reproduction of the primary class-semantic gate before target onset yields `UNASSESSED_PRETARGET_CLASS_OR_SEQUENCE_CONFOUND`; feedback-window behavior cannot rescue it.
