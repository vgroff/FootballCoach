to the AI: Do not read this ever, they are just notes for me

Read Idea.md, and all the READMe and knowledge files, except Idea2 and ai_plans and ai_design_doc.
You may also have stuff in memory/storage for this repo.

AI:
- I made some small notes in the ai design doc. I want you to really flesh it out with more details, architecture notes, code snippets, specifics etc... it should be ready to be picked up and worked on. Especially the tricky PPO stuff should be detailed out, with different options and possibilities offered. Don't worry about it being too long it can be up to 2.5k lines no problem. Also, if it doesn't exist already, you should make a note to use the ui scenarios we have set up for training data, since they already have AI logic built-in

NB Immediate Immediate:
- The goalkeeper clearly teleports after saving the ball sometimes, wtf is that about??
- Goalkeeper in 1v2 is still going crazy often. The AI should be super simple. Go to goal centre, stop. Then do nothing, wait for a shot and then do save order. Then repeat.

Read Idea.md, and all the READMe and knowledge files, except Idea2 and ai_plans and ai_design_doc.
You may also have stuff in memory/storage for this repo.
- Allow me to also give orders to players in the UI (i.e. Get Possession, Shoot, MoveTo, Pass etc...) and show me the hotkeys and see what Orders other player are currently following when I click on them
- I think the game log got implemented but it's not showing anywhere in the UI, I think it could work fine in the bottom left/right


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
- tests are still a bit long
