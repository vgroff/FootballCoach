to the AI: Do not read this ever, they are just notes for me

Read Idea.md, and all the READMe and knowledge files, except Idea2 and ai_plans and ai_design_doc and the other ai-related ones.
You may also have stuff in memory/storage for this repo.

I'm going to give another agent the following task, empty context:
Read ai_trainer_knowledge.md, and then run a training run.

I want you to add 200-400 words of prompt for agent that will carry out more training, give extra detail on the current situation/problems, recent changes, debug logs you've added, things to look out for, lines of investigation etc... Say that the trainer knowledge file needs to be read first thing. Also state that the demonstration needs re-doing, and give the relevant terminal commands to run both that and training

Read Idea.md, and all the READMe and knowledge files, except Idea2. Read the ai_plan.md, the ai_design_document.md and the ai_trainer_knowledge.md

Immedaite task!!
Read ai_trainer_knowledge.md, and then run a training run.
- We could super easliy remove a whole axis of symmetry by transforming all AI positions to being +x is attacking goal, -x is dending goal, remove the flag for which end youre attacking on and transform back and forth from the engine positions to the AI positions. Would probably mean a smaller/smarter AI and faster trainnig from fewer augmentations (half the training time)
    - I think all that needs changing is absolute coordinates - i.e. player position and move region (?). and then augmentation and team flags removed.
- ADD SPEED TO THE PHASE 1 REWARD! ball progress scale/winning should be scaled by time? or at least have a timed component to it? maybe normalised by ball distance to player/opponent goal?
- definitely need a phase 0 type thing where we just teach movement, with and without the ball, going between waypoints for a start, it struggle with movement a lot
- how could I speed up training? parrallelise the engine itself? increase the tick time when nothing much is going on (e.g. ball controlled and nobody near controller?) what is most likely taking up time in this?
- can i forcesome smart exploration by e.g. making the neural net tackle when it's legal and making it learn to do?
- parralelise the simulation somehow? player decision? maybe even movements/updates, though potentially dangerous
- Is the AI completely ignoring all the kick mechanics like direction and spin and just aimming for the centre of the box when it kicks? Why on earth was this implemented???

Immediate AI stuff
- Can we do some trainings runs and then see and see if the trained AI is able to beat the rules-based AI more consistently or at least at a simiar rate than another rules-based AI could?
- watch the 1v1s
- Add training to the UI - let me execute training runs against various phases and let me watch/evalute scenarios entirely in the UI
    - let me also create my own scenarios and play them as training "demonstrations" entirely in the UI, using the actions/orders and having that recorded, and then it gets used for training. store the orders too for neural network training and replaying
    - Allow me to replay the demonstrations in the UI - either through orders or through actions
- during the demonstration collection, print summary stats of the reward scores for each player (they might sum to 0)
- Think about how GetPossesion, Tackle and Move orders are going to interact - they could work together. The Orders would need to break down into AI suggestions maybe

Next immediate training:
- I've made some small changes to ai_plan regarding the movement orders neurons (region of play and importance etc...), can we update the code and documents to reflect this change
- Have Phase 0 training - just teach to follow MoveOrders and RegionOfPlay Order and also both simultaneously with the various HoldPosition/RegionOfPlayImportance values randomised at the start of each run and using varying size and position of regions for both. Start at a random position with random velocity. Only train the execution network on this, but set the order correctly in the input. Use a rules-based AI with MoveOrders to pre-train, have it evaluate the optimal reward function to aim for, and then just select a random point in the target region, then let the AI see if it can do better. Do waypoint chains of 5 waypoints like in the UI scenario, all waypoints within 15m, and pretrain with the WayPointSprint rules AI
- In Phase 1, we can start giving the AI negative examples, by giving it move orders instead of GetPossession orders - in these scenarios, it is only rewarded for getting close to the move point. We can also chain the two, GetPossession first, follow by a move order do a differnt location - with a different reward function as a result. Or MoveOrder first, then GetPossession. I guess we should build some kind of modular/OOP thing where we can chain the rules based Orders and get their respective reward function if theyve been completed succesfully and loss if not, that way we cna make scenarios easily. Things won't always map nicely, we might want behaviours that are more custom, but its a good start. In later cases we might want to have ORders that don't have a rules-based AI
- Passing training - first fix passing by adding a Pass power multiplier and trying it out in the UI passing scenario

AI stuff:
- Speedup ideas - run the simulation at lower resolution, especially at the beginning, and lower the neutral network decision taking rate
- I feel like the AIs should a property of the players like player.ai = StagedGoalkeeper() or neural_net() rather than what we have. I don't understand why wiring in the player AI is so complicated, can it just be player.ai = AI(), then on player.tick(), AI.getAction() and then apply it, or AI.act(player)

NB Immediate Immediate:
- 2V2 scenario is completely broken, I think GK is on the wrong team, and probbaly other shit. Fix it. Also, the through ball pass is doing an awful job of predicting where the player will be, can we hit the pass harder? Maybe for the scenario, make the second attacker move slower, also, is he running GetPossession during the pass?
- The goalkeeper clearly teleports after saving the ball sometimes, wtf is that about??
- Controlling the ball also makes players static I think probably change that, same as with tackling
- We want a bit more visual excitment. Can we display some icons under/over/on top of a player for 1.2 seconds after he carries out an action like kick, tackle, taking control (do a differenet one for goalies) or switching between moving stances/speeds - you can use the leg, hand(GK), wind, running, idle (U+1F574) and soccer ball icons, or any you think are more appropriate, but please no more than 2 per icons per action. Maybe we can also give people speed lines when theyre going above a certain speed, and flashing red outline when stamina is low.
- Much better kick action UI - 1. start kick action by clicking on already selected player, and point in a direction and give it a length (this is xy direction and power) - show the trajectory and error margins as you do so. 2. once you've clicked to select that, moving the mouse back and forth alters the z direction - keep showing the trajectory, and highlight red the parts of the trajetory that are above goal height. 3. Add left/right and up/down spin by circling the kicking player, and once again show the effect and how the error margins are affected
- Why is the ball.possessed_by still using a player_id?? Do we enforce unique player_ids?? It seems so much easier to use just use the possesed_by field be player type rather than string, ball.possessed_by = kicker, instead of kicker.player_id. Is there an issue with circular references or something? If so, can't it be solved gracefully by refactoring or something? If not, it's okay, it just seems ugly
- Maybe slightly reduce the size of the player spheres?
- Is the game log permanent and can I copy/paste from it?

NB Immediate:
- Goalkeeper snaps in save order - not sure its needed if we program things correctly
- what's the point of actions.py? what does it do that orders don't do directly?
- Have we implemented throw-ins, corners and goal kicks? How is the AI going to understand them? Maybe a separate "positioning" AI that positions the players of your team (legally) and then another one decides running direction/speed of the kicker, and the kicker kicks immediately basically
- On the top left/right corner, have a smallish view showing the z axis, just with the ball, the floor and the goals present, no players or anything
- Control times make the game look janky and the ball teleports I think? interpolate moving the ball over to it's final position during control so that things keep moving. What are the players kinematics during control, do they stop entirely? If they slow down, is it organic or snapped? I feel like they should be able to move, just a lot slower max speed, but everything should be organic, no snapping



NB:
- currently stamina regen is faster than sprint depeltion, do we want that?
- Pretty sure you can currently kick at full power with having full in the kick_power stat, which begs the question of what it's purpose is. I think you should only have full power at 1.0 kick power
    - Ask the AI to explain the kick power calculations - maybe we need a non-linearity
- implement goalie rebounds and "failed" saves of various kinds
    - also implement rebounds off players for failed ball controls and from succesful tackles
- Show player attributes ands stamina somewhere on select
- Could do grid search on the goalie intercept maneuver and clever positioning - ask that AI how the intercepting is chosen, but really we should calculate distance/speed for both options, and make a weighted choice based on those, no? throw in some params and re-tune the goalie bonuses probably
- footballer height should vary - and therefore jump height
- At some point ask an AI to: "read through everything - except for now, the ai folder (but do read rules_ai.py) and offer suggestions for refactors or cleanups, duplicate code/logic, string literals, inconsistent documentation/knowledge files, comments and criticisms on structure, easy wins, possible bugs and edge cases, test cases etc... Even if you see something is already explained/documented, if you're not convinced by the explanation or if it still seems dodgy/potentially wrong, bring it up". Do this before extending the game too much beyond the pitch


Past MVP - Late Stage Plans

Mechanics
- Think about introducing non-linearities in kick power, stamina etc... stuff that maybe less spearated in effect than theye are in rarity (gaussian), or the reverse (e.g. dribbling, maybe gets extra good at 80+)

Coaching
- 