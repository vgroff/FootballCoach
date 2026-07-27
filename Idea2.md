to the AI: Do not read this ever, they are just notes for me

Read Idea.md, and all the READMe and knowledge files, except Idea2
We have implemented what is in Idea.md. the following remainds:

More functions:
- Try and improve "pass" by estimating the other players movement and the time to contact and doing small/easy "through balls", maybe just with the other guy jogging and not too far away


New NBs:
- Are players able to stand still again once they've started jogging/sprinting? Especially in the UI is that doable?
- dribbling passed a tackle should still slow you down, depending how well the skill check went (e.g. if your roll was (relative percentage) 35% or more higher, it doesn't slow you down), if you beat it by 0.1% it slows you down 80%. Add balance tests for this
- I think if 2 players do the collision thing and one has the ball and they're going in opposite directions (I guess cosine similarity positive on velocities or something like that), a tackle is automatically triggered instead of the collision calculation. it's not realistic to have players charging each other frontally haha
- goalkeeper tackling checks get +100% rather than the usual +20%
- I want to change the tackle order: it should now be called "get possession" where player just runs straight at the ball, it falls back to what is now the tackle order if someone else has it but if they don't he just tries to get posession



NB:
- currently stamina regen is faster than sprint depeltion, do we want that?
