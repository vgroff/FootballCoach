to the AI: Do not read this ever, they are just notes for me

Read Idea.md, and all the READMe and knowledge files, except Idea2 and ai_plans.
You may also have stuff in memory/storage for this repo. We now want to add some more features to the repo. the plan has already been made in current_plan, so read that. We have done phases A-F, now implement Phases G and H including the test

Read Idea.md, and all the READMe and knowledge files, except Idea2 and ai_plans.
You may also have stuff in memory/storage for this repo. We now want to add some more features to the repo. They are detailed below. Read the files you need, ask questions etc... Write me a high-level design document detailing what needs to be done, and which files are useful to read for each task and in general. It will be used by other agents. Also, detail the maths and numbers and say why you choose certain parameters or models, I'd like all that stuff and I like to understand those kinds of mechanics


- In the move to order, avoid nearby players (have a kind of repulsion mechanic with distance for players < 4m away or smth) - stronger repulsion if you have the ball that makes you slow down, no repulsion if the nearby player in question has the ball. Do a dot product between the relative velocity and the repulsion - if they're too aligned, add adjust move direction in the orthogonal direction to the repulsion by some minimum value. Then also add balance tests that confirm that a decent player will run straight into another player when the repulsion and mimimum direction adjustment are set to 0 and that they will run around correctly when it is set to the right value (and we have to find this value)
- penalise goalkeeper tackles when they do it outside their box (by 40% for now to discourage them). Add tests+balance tests
- Goalkeeper save - shouldn't just try to do the save on the goal line - if ball is near and we think we can intercept it early then intercept it early rather than on the goal line - write tests for this, we just don't want false positives or performance degradation. I guess it will be using the same logic/code as "Get possession" order does
- Goalkeepers "jumping" - goalkeepers shoudl be able intercept shots taller than them (if this isn't already the case), but it should be extra penalised with the ball control delay thing when that's the case, and it should depend on how high it is. add balance tests and regular tests
- We should visually show both inactivity and ball control delay on the UI - with an outline I would think
- If a player is tackled while they have the ball but they are currently in the process of controlling it, their dribbling roll on the skill check is penalised by between 0% to 25% depending on how long they have left on the control timer
- Player collisions should reduce speed significantly along the axis of collision, even if a tackle is triggered. I think we get unrealistic physics otherwise. Test it once implemented
- UI Scenarios:
    - Only the shooting one (Close range mixed results), the passing one, the tackling one and the running one are that useful. Remove the rest, only have those 4
    - One the shooting one: allow me to set some of the more interesting parameters that go into the randomness, like player positions range, kick precision range, speed etc... whichever ones are gonna be interesting, not necesarily every single random number, just the main 5-10
    - On the tackling one: on each run, randomise the positions of each player, keep them within 10m of each other and have the attacker jogging in a random direction
    - On the running one: on each run, make a random course - i.e. 5 points that are 5-25m away from each other in a random direction (but never going off the field) and have a decent runner involved too
    - On the passing one: randomise the distance and position of the players, but always have them within 30 metres, also have them be somewhat high attributes (70-80s) if they arent already
- Running direction should affect shot precision - do a cosine sim between running and shot directions, then if it's above above 0.35, there is no effect, between 0.35 and -0.2 it scales from 0 to -25% on shot precision stat and then from -0.2 to -1 it scales from -25% to -60%. Add tests/balance tests for this behaviour
- Have a visible indicator of the ball being in a rolling state, flying state and a "just bounced" state, I'm thinking a thin outline of some kind
- Add a game log - bottom corner somewhere, just some a scrolling window with messages evertime: a player takes possession, a kick happens (with params displayed), a goal is scored, or an order is given (with params). We probably we will want to add different levels of logging (behind a flag so that its skipped in tests, but always enabled for the UI) that we can switch between in the window so that we can see more detailed info if needed.
- I'd be interested in trying rudimentary 2v2 scenarios in the UI - 2 attackers, 1 defender+1 GK, one has the ball and is in the box aligned with the left post, the other is slightly behind in x (to stay onside) and aligned with the right post but is running forwards. The player with the ball passes to the other, and the other either shoots immediately or controls it first, moves slightly forward and then shoots. Everything is carried out using the Pass/Move/Shoot orders for attackers, while the defender runs GetPosessions and the GK runs save.
- I want to implement a Mark order - the player should basically stay close to a targeted opposition player and stand between him and the ball, and then try to intercept the ball if it gets close (use GetPossession code/logic for this). If his target has possession, he can also just fallback to using GetPossession code/logic to tackle him.
- Let all scenarios run for an extra 2 seconds after theyre detected as completed

NB Immediate:
- Goalkeeper snaps in save order - not sure its needed if we program things correctly
- what's the point of actions.py? what does it do that orders don't do directly?
- Have we implemented throw-ins, corners and goal kicks? How is the AI going to understand them? Maybe a separate "positioning" AI that positions the players of your team (legally) and then another one decides running direction/speed of the kicker, and the kicker kicks immediately basically


NB:
- currently stamina regen is faster than sprint depeltion, do we want that?
- Pretty sure you can currently kick at full power with having full in the kick_power stat, which begs the question of what it's purpose is. I think you should only have full power at 1.0 kick power
    - Ask the AI to explain the kick power calculations - maybe we need a non-linearity
- implement goalie rebounds and "failed" saves of various kinds
    - also implement rebounds off players for failed ball controls
- Show player attributes ands stamina somewhere on select
- Could do grid search on the goalie intercept maneuver and clever positioning - ask that AI how the intercepting is chosen, but really we should calculate distance/speed for both options, and make a weighted choice based on those, no? throw in some params and re-tune the goalie bonuses probably
