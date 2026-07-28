to the AI: Do not read this ever, they are just notes for me

Read Idea.md, and all the READMe and knowledge files, except Idea2
We have implemented what is in Idea.md. the following remainds:

More functions:
- Try and improve "pass" by estimating the other players movement and the time to contact and doing small/easy "through balls", maybe just with the other guy jogging and not too far away
- fleshing out tackling - angle of the tackle should give bonuses and especially maluses when e.g. tackling from behind (i'm thinking add another modifier to the tackling roll, +10% for a fully frontal tackle, down to -5% for a 90 degree tackle, -65% for a fully 180 degree tackle)
- when running at the ball during "get possession", try to run ahead of it (or the player) by estimating where it will be using the velocity and yours (do a rough approximation)
- This is no pass action! Only kicks. I'd rather understand why kick isn't working well as a pass than fake something like that

New NBs:
- In the move to order, avoid nearby players (have a kind of repulsion mechanic with distance for players < 4m away or smth) - stronger repulsion if you have the ball that makes you slow down, no repulsion if the nearby player in question has the ball. Do a dot product between the relative velocity and the repulsion - if they're too aligned, add adjust move direction in the orthogonal direction to the repulsion by some minimum value. Then also add balance tests that confirm that a decent player will run straight into another player when the repulsion and mimimum direction adjustment are set to 0 and that they will run around correctly when it is set to the right value (and we have to find this value)
- penalise goalkeeper tackles when they do it outside their box (by 35% for now to discourage them)
- in the ui allow me to load in a generic shooting test similar to some of the existing ones but with full control over the boundaries of the random numbers like positions and kick precision, whichever ones are gonna be interesting, not necesarily every single random number
- Goalkeeper save - shouldn't just try to do the save on the goal line - if ball is near and we think we can intercept it early then go for it - just write tests for this, we don't want false positives
- Use the goalkeeper positioning we've implemented in the UI scenario for tuning gk values in the balance tests

NB:
- currently stamina regen is faster than sprint depeltion, do we want that?
- Pretty sure you can currently kick at full power with having full in the kick_power stat, which begs the question of what it's purpose is. I think you should only have full power at 1.0 kick power
- implement goalie rebounds

