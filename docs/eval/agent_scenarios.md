| Scenario | Expected behaviour | Pass | Corrections | Plan changes | Questions |
|---|---|---|---|---|---|
| good form, whole set | no corrections, only encouragement | ✅ | 0 | 0 | 0 |
| one-frame glitches (noise) | single-frame faults are ignored (no false alarm) | ✅ | 0 | 0 | 0 |
| hips sag, user fixes after cue | cue spoken, re-check confirms the fix, praise | ✅ | 1 | 0 | 0 |
| hips sag, user never fixes | cue, repeat, then the agent changes the plan | ✅ | 4 | 2 | 0 |
| knees drop repeatedly | switches to knee plank; knees on floor stop being a fault | ✅ | 2 | 1 | 0 |
| user stops after 20 s, comes back | timer paused, 'you held 20 s', resumes by itself; paused time not counted | ✅ | 0 | 0 | 1 |
| user stops, chooses 'rest' | agent asks continue/rest; the human decides to rest | ✅ | 0 | 0 | 1 |
| user stops, no answer, never comes back | after 20 s paused, the set ends as rest (safe default) | ✅ | 0 | 0 | 1 |
| question unanswered while still exercising | escalation never blocks: plan change or question | ✅ | 6 | 2 | 0 |
| Bedrock unavailable | fallback policy decides, no crash | ✅ | - | - | - |
