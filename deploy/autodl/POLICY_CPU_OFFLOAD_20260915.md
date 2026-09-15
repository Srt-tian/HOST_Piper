# Policy restart: CPU optimizer offload

User explicitly requested cause analysis, lower per-GPU load and resubmission on the same
AutoDL4xA80080GB instance. No new instance, no EIP task, no alignment retraining.

Prior formalrun stopped at02:25 afterupdate3: rank3reservedpeak74.27GiB>74GiBguard,
not CUDAOOM. Other ranks showed allocatedpeak~60GiB/current~41GiB while reservedgrew
~68->74GiB. Idle cache/fragmentation is implicated but failing-rank allocatedmemory was
not recorded because the guard raised before logging. Fix recordshealth BEFORE raising.
Only original2-step probe weights survive; formal3updates were never saved.

Cannot reduce microbatchbelow1. New policy_zero2_cpu.json offloadsFP32master/Adamstates
and optimizerstep toCPU using DeepSpeedCPUAdam; gradientcheckpointing staysenabled.
CPUAdam actualtinyupdate passed in isolatedpolicycache. Native ninja executable copied
into policy_runtime/bin from existing alignmentruntime; sourceenvironment unchanged.
No base packages changed, no data/weight downloads. Expected GPUreduction~24GiB/rank
for8.5Bparameters/4shards (FP32master+m+v); actualpeak must be measured.
Existing hostlimit480GiB/112CPU observed; CPUoffload consumes hostRAM/PCIebandwidth.

New task=piper_autodl_cpu preserves micro1,accum4/global16,three224px views,30nativefuture
executed-state actions,DTW1500labels,realT5,velocityobjectives and lr1e-5/visual10x.
1000updates,50warmup,saveat1and5/every20/final,model-onlykeep2,diskreserve8GiB.
64GiBreservedguard is LOWER thanold74GiB; do not silently raise it to pass a test.

Diagnostics: perrank/microbatch samplepaths,action/proprio/progressranges,three lossbranches,
noise timesteps. Aggregateallmicrobatches/allranks with sampleweighting intometrics.jsonl;
legacy component display remains lastmicrobatch where not explicitlywindowaggregated.
No autogradgraphs retained inmetrics. Resetpeaks after each update afterhealthlogging.
Velocitytarget isnoise-sample and timeweight isbounded; no1/sigma train-targetdivision.
The earliernorm5007.93 spike ispreclip (clip1.0), finite, and not its own failurecause.
Exactsample/branchsource remains unknown pending newdiagnostics; do not inventa cause.

Use existing2-step seedcheckpoint with strict five-module loading. Transfer its singlefile
into the new run's initialrollingcheckpoint namespace rather than retaining an extra16GiB
copy elsewhere. Preserve previouslogs/health/config and latestalignment1400/1500/rawdata.
Verify at least5updates (pastoldfailurepoint3) and actualearlycheckpointpublication before
claiming the restart is stable; not a guarantee against all later failures.
