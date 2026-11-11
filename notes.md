pd-arena:
    - rollout lengths are 50. Might need to retry with smaller rollout lengths.

Recurrent MAPPO:
    - GRRRRR I used a GRU instead of LSTM... Might need to rerun experiments if needed. However, I'll leave it for now. 


Recurrent IPPO:
    - Do I want the critic or actor predict the location. The most logical thing would be the actor as it would force it to link good positions to high value.
    It matters less for the critic as all it cares about is its memory and the current view. Not per-se location. I will only add it to actor for now. 
    Later it might be interesting to see what happens hwen we force the hidden state of the actor to bethe same as that of the critic. Or that would just mean
    that you have one body with two heads so that is not that usefull though.