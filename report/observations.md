## Observations 12-01
### Angular momentum constraint calc

### System specific changes

### Solver not working pal!!!
 - *Observation*: controls and costs blowing up at jump-aerial phase switch. 
 - Diagnosis: Logging showed feet start to slip at the take off moment. Phase switch detection was delayed, as a result phase 1 controller was working when phase 2 was active.
 - Solution: For phase switch: Tightened the bounds on the feet positiion changed from 1e-2 to 0, and the time based condition from $1.1*jump time$ to $0.9*jump time$. For controls ubruptly blowing up: Introduced exponential averaging of the controls. This also greatly aided the abrupt increase in body angular momentum.



## Landing phase fix
### Branch: ablation-studies, commit: 8ac523b, Anirudh's PC
 - *Observation:* Noticed during the tracking of the feet position during flight phase, one foot was tracking well, the other fails to track after a while and the costs drastically increased at that moment. 
 - *Diagnosis:* One foot was landed, and not the other.
 - *Solution:* I changed the phase switch definition to any of the foot in contact instead of both. 

## Adjusting gains for torso landing and feet

### Logs: 12-01_23-17 to 12-01_23-54

### Logs: 12-01_23-17 to 12-01_23-54, Branch: ablation-studies, commit: 449ed1f..4d48023, Anirudh's PC

Idea: Torso angle and torso posotion graph animation

## Trying out contact implicit OSC

 