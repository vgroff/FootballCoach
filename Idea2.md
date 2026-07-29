to the AI: Do not read this ever, they are just notes for me

Read Idea.md, and all the READMe and knowledge files, except Idea2 and ai_plans and ai_design_doc and the other ai-related ones.
You may also have stuff in memory/storage for this repo.

AI stuff:
- are the best models saved? In the UI scenario, have a checkbox for each player that allowed me to have them controlled by the best available AI instead of by the rules-based AI.
- Speedup ideas - run the simulation at lower resolution, especially at the beginning, and lower the neutral network decision taking rate
- Git ignore the checkpoints and shit

NB Immediate Immediate:
- The goalkeeper clearly teleports after saving the ball sometimes, wtf is that about??
- Goalkeeper in 1v2 is still going crazy often. The AI should be super simple. Go to goal centre, stop. Then do nothing, wait for a shot and then do save order. Then repeat once save order terminated. Explain the AI to me.
- Tackles seem to behave a bit weird, I think the players often stop almost completely, and while I think speed should be reduced somewhat for both players, it's maybe too strong, also I think collisions should be turned off once a tackle has been engaged, I think that is supposed to happen with one of them going inactive but I just want to make sure the collision isn't being triggered on the same tick as the tackle either way.
- We want a bit more visual excitment. Can we display some icons under/over/on top of a player for 1.2 seconds after he carries out an action like kick, tackle, or switching between moving stances/speeds - you can use the leg, wind, running game, idle (U+1F574) and soccer ball icons, or any you think are more appropriate, but please no more than 2 per icons per action.
- Why is the ball.possessed_by still using a player_id?? Do we enforce unique player_ids?? It seems so much easier to use just use the possesed_by field be player type rather than string, ball.possessed_by = kicker, instead of kicker.player_id. Is there an issue with circular references or something? If so, can't it be solved gracefully by refactoring or something? If not, it's okay, it just seems ugly
- Maybe slightly reduce the size of the player spheres?

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
    - also implement rebounds off players for failed ball controls
- Show player attributes ands stamina somewhere on select
- Could do grid search on the goalie intercept maneuver and clever positioning - ask that AI how the intercepting is chosen, but really we should calculate distance/speed for both options, and make a weighted choice based on those, no? throw in some params and re-tune the goalie bonuses probably
- footballer heigh should vary - and therefore jump height
