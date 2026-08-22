to the AI: Do not read this ever, they are just notes for me

Read Idea.md, and all the READMe and knowledge files, except Idea2 and ai_plans and ai_design_doc and the other ai-related ones.
You may also have stuff in memory/storage for this repo.

I'm going to give another agent the following task, empty context:
Read ai_trainer_knowledge.md, and then run a training run.

I want you to add 200-400 words of prompt for agent that will carry out more training, give extra detail on the current situation/problems, recent changes, debug logs you've added, things to look out for, lines of investigation etc... Say that the trainer knowledge file needs to be read first thing. Also state that the demonstration needs re-doing, and give the relevant terminal commands to run both that and training

Read Idea.md, and all the READMe and knowledge files, except Idea2. Read the ai_plan.md, the ai_design_document.md and the ai_trainer_knowledge.md


Extending training:
- We have a big system in place for Phase 1 training. a bit part of this task will be generalising this in intelligent ways for multi-task training.
- Have a look at the balance scenarios in the UI. I want to use some of these as inspiration for tasks. In particular, at first, the passing one and the close range shooting and 1v2, to teach kicking/passing/shooting/scoring. 
- Essentially we wnat the exact same system, but with different rules-based AIs and reward systems
- I'm thinking the reward function should be split by order. E.g. for phase 1 we have these custom rewards that are relaly just tied to Order behaviour. Instead, it should be split by order - eg some apply to GetPossession/ChaseTackle (depends on ball possession), some apply only to MoveOrder, then later some will apply only to Shoot, Pass etc… completing the order (fast) should always give a reward boost. This was we can build rules-based AIs, set the appropriate order and then query them to see what the reward should be, both for themsleves and for the neural AI. There will also certainly be custom reward termsinvolved in each case, some we cna make always be there (e.g. stamina)
- It sounds like reward functions are currently personalised per-player. They should be generalised and apply to all players with a particular role in the task (reward may change by role)
- Task ids are important, make sure they are set
    - I think task ids should only be set by the decision network, the execution network shouldnt get them (remove if it already does)
- Certain coefficients may need setting on a per-phase level, hopefully not too many, but potentially things like BC aux (e.g. kicking only or not)
- Record demonstrations needs to handle more than 2 players
    - Sould be able to do some nice OOP here, a class that has the player in question and it's rewards and outcomes, or whatever, let's make it clean
    - Might be some useful notes in agent_plans/reward_fixes.md if it hasn't already been implemented
    - Should also apply to the diagnotics during training, should be able to count outcomes/rewards per-player and print them sensibly 
Notes to self:
- GK needs fixing - he saves no shots in the close rnage scenario
- When running these messy/random scenarios, we might need to decide (maye using a value network) whcih runs are actually good (advantage-wise and in absolute)

Since the previous request explicitly said "scenarios end with: win / loss / ballout / timeout — THAT'S IT, nothing else". Is that what it said though? Or are you intentionally paraphrasing? did it specifically have "ballout (split by intentional and not intentional)" per chance? OH look, it did. Maybe you should've clarified if you were unsure



Current notes:
- " [task] : read ai_trainer_knoweldge.md, ai_config.json and training_Runs.log entirely. Please do not skip any of them. what do we think of how the training is going? "
- " [task] : read knowledge.md, ai/knowledge.md and ai_trainer_knowledge.md entirely. Please do not skip any of them. "
- train blockers:
    - !! Broken current order stuff AI has found bugs!!
    - !! real neural net: don't normalise by pitch half diag, just normalise by base pitch half diag
    - !! real neural net: ball spin should be normalised by a fix value rather than this random ass max_rad thing
    - !! physics network
        - test quality of embedding with a small capacity (e.g. 10 neurons), newly initialised decoder + shortut switched off - how quickly can it learn? especially during demo pre-train
            - do tiny batch size for new ones, huge batch for old ones than need fine tune
        - try switching spin and bc weights back on - do we care though? Time/position of ball out could be useful though. could then go as an input for the proper network
        - consider - a tiny decoder trained to output the positions/velocities at some time horizons - then tacked on to the output in the proper network
            - maybe also have it predict time and crossing point for going out
        - Cooked idea but could be fun for later - take the ball encoder and player encoder, run trials where both move and train a decoder (tiny hidden layer, like 8 or smth) on top of both encoders to predict player-ball distance at some future time t (e.g. 0.2 and 1s and 4s), and then you can change the move order in the player encoder, and "see" player-ball distance in the future. Could then be a learnable parameter for the (self-player) policy network to see what different move orders would do, and act on that basis/affect the attention etc...
    - !! generate a ton of demo data uisng vvgood immobile
        - !!!!! Does it have a random seed each time???
        - for value net pretraining + BC when we change a neural net
        - TIMESTEPS ARNE'T UNIFORM!! - how does MC deal with this?

        - on debug value net - check the match logs that do poorly on invalid
    - !!!!!!! Add heading to the neural network. Also increase capacity, and do BC training I guess
        - !!! heading add by just making velocity + speed a unit vector + magnitude?
    - !! Implement and try out the physics encoder thingy, just for the ball
    - In demo generation - encode empty players more intelligently to use less RAM? float 16 for the decision heads (marginal)?
    - !! Remove ball height norm so everything is euclidean again?
    - !!! Do some performance profiling - how long spent on rollouts vs PPO vs evals etc.. n_envs - how much faster does it get with more of them? How many do we want? Main threads too? - can try this with the evals script maybe?
        - vary opponents!
        - make outcomes from the POV of the player - especially for reporting stats, or it'll break. Currently outcomes are global (e.g. win is for trainee win)
    - !!! Phone note:
        - Shorter timeout so that we get a bit more of them?
        - Does lowering dt improve performance or not?
    - try annealing? - worth proving that kick push is actually faster tbh
    - !! value network HAS to work on rules-based, used the debug value net script to prove it - there's almost no variance per-seed (Claude showed this will the evalute script in the "Review PPO..." chat)
    - Is any data missing during BC or does it have everything? If so, why can't it predit it corectly?
        - boost sprint/move loss??
    - !! Do a full-ass BC training all the way
    - !! value pretrain - why does it do so much better during PPO than during pretrain??? is the reported loss before or after re-train? Can we get the other one too? Is it using the right decision network in pretrain?
        - what if we load in the actual network in the debug value net script? deterministic vs not? Can it do as well/better?
        - what's frozen/shared for the value network?
        - try debug value net with the decision network heads insted of it's own network
    - Demo should log kick aim, not acutl kick direction (post-error)
    - make running speed continuous - could help with the dt quantisation, and easy to do (don't change rules-based)
    - log entropy + entropy loss contribution for each head
    - print what the actual MC contribution of each rewrd is per step
    - Try more aggresive learning just to see how it does? 
        - Increase step penalty?
        - Maybe re-introduce the shaping rewards?
    - DO players have current stamina?
    - bring annealing back?
    - re-introduce speed bonus or nah?
    - Plan: 
        - need to re-run with the reset log dir std?
        - do we need more data?
            - try with non-separate value net
                - double chek everything is sensible with frozen layers and parameters etc... maybe switch to using the decision net head rather than execution? maybe test it on a pretrained model vs sepearte ones. would make more sense though
    - [] !! does BC do as well as rules AI? No reason it shouldn't
        - DOES DEMO DATA RECORD MOVE SPEED TRANSITIONS /DO THEY ALWAYS HAPPEN ON DECISION TICK?
            - What should I set the demo_sample_interval_s so that every rules-based AI decision is logged and trained on? How often 
        - try longer BC, and/or more data
        - force an overfit, and see if it can replay the games exactly. demo_sample_interval_s might need changing
    - Do anti-antilaising for the spots on the ball, also the anti-aliasing on the players seems a bit broken, do they have an outline or smth? Also, I'm not sure either the ball outline or the state rings are anti-aliased. Also also, how does the width work with those, and how does the offset work on the rings?
    - not 100% clear on why we need the attention heads, since the entities are all equivalent, could we not just jack them straight into the trunk with shared weights? just a thought, not sure if it's actually a good idea tbh, just not clear - if the purpose of attention is to maintain some location differnece between them is it needed? I'm probably bieng dumb
    - does phase 0 do as well as the debug script? - no reason it shouldnt really
    - maybe boost value network capacity a little
    - try annealing again?
    - Do some tests with inline code and small rollout to see which many envs/worker combinastions are fastest, and how much capacity matters
    - Can value network predict with only immobiles? Should be easy AF
    - rules-based AI should reposition between enemy and the goal if they are slower and it's possible. also, try to tackle from the front if possible
    - !Use the debug script to check how the torch threads affects training and change the defaults correspondingly if needed (including for PPO!) - I get the sense it helps, but not 100% sure
    - value loss kinda sucks - maybe because of kicking? maybe because it should be one network?
        - step back for a sec - if it can't predict the rules-based AI rewards, then we're fucked (tbf, they use rules v rules, so maybe see how it does on rules vs immobile). Check which ones it struggles with and what the numbers are actually like. Is it predicting the monte-carlo stuff as it should be?
        - to a train/test set for val loss on rules-based demonstrations and see how it does
        - Get the match logs of the worst predictions in the val set
    - The UI Phase 1 (and maybe all?) should use the same scenario thingy code as training, and it should use that to output rewards in the game logs when they trigger (e.g. get_possesion +1 for trainee) when showing scenarios. Makes debugging easier
    - BC train, then do deterministic vs immobile and see how we do compared with PPO
    - always have the option of doing a small set to scenarios each time, say 100, so that we can really track progression?
    - [] value netowrk fixed - can we switch back to not having a separate value network?
    - [] still getting very large policy ratios sometimes - check what is the contributing factor (Dont we already see this?)
    - [] get match logs for best performing (by reward) and "smartest" (most adv, idk) runs each epoch
        - [] make a match log viewer (static image with lines)
        - [] anytime a rewards gets clamped - output the match log and examine why
    - [] check speed bonus works correctly
    - [] some useful stuff from chatgpt
    - make a giant BC training set (background task)
    - [] no illegal move penalty - sus
    - Do the seeding thing somehow, or have pre-built scenarios, or something like that. Or maybe use the same exact scenario(s) but add a bit of noise to everything
        - [?] Could already do this by just having very tight params? Maybe add some more bounds first?
    - Policy ratios are "insane". Investigate, very possible from sigma being small
    - "close miss" type penalty - distance to ball increasing while small while ball not possessed
    - Think about NN inputs a bit more - Should we give relative velocity/speeds to players/ball or similar? Giving relative speed might be a compromise
        - Self/other distinction - feed self through the entity MLP + attention, but give it a separate MLP just for the querying on the second order, and then no self_mlp feeds into the trunk
        - Keys and values use the same embedding (other_embed feeds both). nn.MultiheadAttention does have internal separate W_K and W_V projections, so it's not as bad as it sounds — but giving keys and values explicitly different upstream MLP branches would let keys be tuned purely for relevance scoring while values carry richer information.
        - Single attention layer is the biggest limitation. You can't model player-player interactions — e.g. "player A is relevant because they're near player B who has the ball". Stacking two transformer layers would give second-order interactions but is a meaningful architecture change.
        - Consider adding the intermediate attention to the trunk?
        - The shared per_entity_mlp for self (query input) and others (key/value input) forces both to live in the same embedding space. Separate MLPs for self vs others is common in entity-based architectures and would give more freedom.
=====================
    - We have a plan for the implementation of seeding scenarios for less noise
        - The evaluation steps should also use seeds so that they're always the same!
            - might need multiple runs of each trial to reduce noise better?
        - For now, only implement on the evaluations during training and before training, make them all the same (might need to harmonise number of trials param)
    - "Value only continuation" should use a test set with early stopping (generate new ones, 10% size comparison)
    - give the football a black border/outline in the UI - expose the thickness
    - Does neural AI kick power adjust for running speed or not?
    - rules-based AI kicking behaviour seems weird, both doing it when it shouldnt and not doing it when it should it seems - is it calculating speed vs top speed correctly?
        - Prove rules-AI performs better with it
    - Reward shaping rewards should anneal to 0 over training - only leave speed and victory and timeout/ballout type ones
    - can rules-based AI run at 2Hz? I.e. only make move/actions changes on decision tick. Does it perform similarly well?
    - have better/more varied move/kick/tackle data, it seems to confuse them - kicks and tackles are random, as are player attriutes, so we should run have the same game scanerio multiple times in the BC to show the difference!!!!! Should we do this during PPO also?
        - Should we even just have like 1 scenario? or a small selection? use a set of random seeds? E.g. seed 1-10, always the same 10 scenarios
        - Different ones for positions vs attributes vs match seed etc? One set of seed for position, another for attributed (or randomise it)
    - do we/should we punish the network for kicking/tackling when it cant or just ignore it?
    - what are the illegal decisions in the reward function?
    - why is the BC network unable to relialy learn tackling? It seems sus, we should inspect it's inputs when it does and doesn't predict a tackle correctly
    - getting 70% against immobile - check in the UI maybe
        - Much worse performance again immobile than self-play - it just runs in ncircles in the UI -  why??
    - do we need batch norm or dropout oor other regularisation?
- what would happen if the correlations in the attriute matrix made them an invalid gaussian type thing?
- 2v2 goalkeepr AI should only switch to save order if the kick is aimed towards the goal, and should cease save order if the ball changes posssesion, even to another teammate of the oppoiste side, and go back to the goal centre


Other todo
- Reduce decision rate everywhere (0.4s?)
- Which tests are failing and why? What are they testing?
- PRobably need a small reworrk on spin, how its' affected by talent and max values
    -have an agent plan on this


Immedaite task!!
Read ai_trainer_knowledge.md, and then run a training run
- in some cases it might make sense to feed in more "real" values, e.g. shot error or top speed or acceleration, rather than the player attributes. that way we can fuck with the physics without losong AI kowledge
- We could super easliy remove a whole axis of symmetry by transforming all AI positions to being +x is attacking goal, -x is dending goal, remove the flag for which end youre attacking on and transform back and forth from the engine positions to the AI positions. Would probably mean a smaller/smarter AI and faster trainnig from fewer augmentations (half the training time)
    - I think all that needs changing is absolute coordinates - i.e. player position and move region (?). and then augmentation and team flags removed.
- definitely need a phase 0 type thing where we just teach movement, with and without the ball, going between waypoints for a start, it struggle with movement a lot
- do we need a von mises on angle stds? or is the gaussian good enough?
- how could I speed up training? parrallelise the engine itself? increase the tick time when nothing much is going on (e.g. ball controlled and nobody near controller?) what is most likely taking up time in this?
- can i forcesome smart exploration by e.g. making the neural net tackle when it's legal and making it learn to do? This is called DAgger, but we already kinda do it with BC annealing, it's just online instead of saved data - but we cna do it later
- parralelise the simulation somehow? player decision? maybe even movements/updates, though potentially dangerous
- Simplify match logs - only keep possession, kicks, tackles (incl. attempted), goals etc... and the positions and time of the event
- Use von Mises distribution for direction
- Scenarios to run for rules based tests (and later NN phases)
    - Intercepting the ball - does it work well or do we need a higher safety threshold
    - 2v3 (with GK) (or even 2v2) test for rules based marking and through ballstate

Immediate AI stuff
- Add training to the UI - let me execute training runs against various phases and let me watch/evalute scenarios entirely in the UI
    - let me also create my own scenarios and play them as training "demonstrations" entirely in the UI, using the actions/orders and having that recorded, and then it gets used for training. store the orders too for neural network training and replaying
    - Allow me to replay the demonstrations in the UI - either through orders or through actions
- Think about how GetPossesion, Tackle and Move orders are going to interact - they could work together. The Orders would need to break down into AI suggestions maybe

Next immediate training:
- Have Phase 0 training - just teach to follow MoveOrders and RegionOfPlay Order and also both simultaneously with the various HoldPosition/RegionOfPlayImportance values randomised at the start of each run and using varying size and position of regions for both. Start at a random position with random velocity. Only train the execution network on this, but set the order correctly in the input. Use a rules-based AI with MoveOrders to pre-train, have it evaluate the optimal reward function to aim for, and then just select a random point in the target region, then let the AI see if it can do better. Do waypoint chains of 5 waypoints like in the UI scenario, all waypoints within 15m, and pretrain with the WayPointSprint rules AI
- In Phase 1, we can start giving the AI negative examples, by giving it move orders instead of GetPossession orders - in these scenarios, it is only rewarded for getting close to the move point. We can also chain the two, GetPossession first, follow by a move order do a differnt location - with a different reward function as a result. Or MoveOrder first, then GetPossession. I guess we should build some kind of modular/OOP thing where we can chain the rules based Orders and get their respective reward function if theyve been completed succesfully and loss if not, that way we cna make scenarios easily. Things won't always map nicely, we might want behaviours that are more custom, but its a good start. In later cases we might want to have ORders that don't have a rules-based AI
- Passing training - first fix passing by adding a Pass power multiplier and trying it out in the UI passing scenario

NB Immediate Immediate:
- At some point ask an AI to: "read through everything - except for now, the ai folder (but do read rules_ai.py) and offer suggestions for refactors or cleanups, duplicate code/logic, string literals, inconsistent documentation/knowledge files, comments and criticisms on structure, easy wins, possible bugs and edge cases, test cases etc... Even if you see something is already explained/documented, if you're not convinced by the explanation or if it still seems dodgy/potentially wrong, bring it up". Do this before extending the game too much beyond the pitch
    - WE DID THIS! It's in code_analysis, a buch of it is implemented but not all
    - Once, and we should again, with the ai stuff, and also a non-code/physics one on the football side of things, one on the UI side etc..
        - ai stuff - ppo_trainer is massive, does it really need to do all that stuff, is some of it not duplicate code? Can some of it be refactored elsewhere?
        - also get them to check knowledge doc correctness
    - check knolwedge doc agreement (between themselves)
- The goalkeeper clearly teleports after saving the ball sometimes, wtf is that about??
- Manual kick-order
    - secondly, we need a better way of showing the z trajectory, I'm thinking 2 little views somewhere showing the z trajectory, especially compared with the goal
- We need neural network spin, plus implement spin error (including for the trajectoris displayed in the ui for the human player during kick order)
    - we have an agent plan for spin
- Why is the ball.possessed_by still using a player_id?? Do we enforce unique player_ids?? It seems so much easier to use just use the possesed_by field be player type rather than string, ball.possessed_by = kicker, instead of kicker.player_id. Is there an issue with circular references or something? If so, can't it be solved gracefully by refactoring or something? If not, it's okay, it just seems ugly
- Maybe slightly reduce the size of the player spheres?
- Is the game log permanent and can I copy/paste from it?
- At some point the details of tackling, auto-tackling, and collisions should be worked out, I'm not sre its all that realistic
- Kicking and running - this should be possible, and faster than running with the ball. Do some tests, and then add it to Phase 1 AI (and/or MoveOrder) 
- kicking direction needs working - can’t kick at 90 degrees upward, even less with the same power
- Allow pausing and going back in time (up to 30s)
- Control - should set the ball to ground level, snap for now but improve it later? Also how long are the control delays vs real life? Also implement failed control - rebounds. What happens during control, does a player have ball possession and can he be tackled while controlling? Shouldn’t be possible until ball on floor
- Heading - should reduce shot power (and precision) 
- Give the ball some dots and make them spin during spinning
- Possible later optimisation - the execution network could have only decisions+latent space as input, with none of the other current inputs present? Or way fewer at least? Would make it run faster, it can be smaller, the larger decision network can run less often

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
- Show player attributes somewhere on select
- Could do grid search on the goalie intercept maneuver and clever positioning - ask that AI how the intercepting is chosen, but really we should calculate distance/speed for both options, and make a weighted choice based on those, no? throw in some params and re-tune the goalie bonuses probably
- footballer height should vary - and therefore jump height
- certain things in physics.json should probably be in attributes.json or a differently names config file, we should probably hve an entire audit of these and figure out how best to organise the files/config params


Past MVP - Late Stage Plans

Mechanics
- Think about introducing non-linearities in kick power, stamina etc... stuff that maybe less spearated in effect than theye are in rarity (gaussian), or the reverse (e.g. dribbling, maybe gets extra good at 80+)j - [Late thing] - Implement a foot choice when kicking, and foot preference and degree of ambitextricity etc... foot choice affects power/precisions depending on direction of kicking vs velocity heading, it affects spin in the same way 

Player stuff:
- Traits
    - "risky" - higher temperature, vs "play it safe" lower temperature
    - 

Matches:
- Could add some commentary kinda easily - e.g. pass happens, say a phrase like "X passes over to Y"

Network architectures to try:
- Try to fuck with attention - more heads and dims, more , different architectures, different things going into
- try adding residual layers on the trunk layers (like layer 1 -> layer 2 and then have a layer 1 + layer 2 block)
- Not sure I understand

Coaching
- Your players train their neural networks over time on actual data
- have standardised “sets” that players learn from - eg football 101 has basic passing and shooting, 102 has more advanced - training data can be randomly generated but have a known test set
    - Includes reinforcing the decision neurons
    - Could have the scenarios created by a network
- Players can remember specific “good” training examples from their career and have their personal data
- You can create your own scenarios in training or from matches, and “correct” bad behaviour
- Maybe allow players to gain more neurons, or faster reaction times (this could break the shared weights computation, unless we approximate)
- Limit on human actions per game - can be increased through coach attributes 
- Mechanics for improving morale
- Some attributes then: training, oratory (for morale), negotiating, 
Managing
- a neural network to calculate player value - have a bidding system for players and then have the teams play against each other
- Each player has a “potential” - e.g. 0.8. Then progress on attributes is determined with a penalty depending on how close to 0.8 (maybe use a sigmoid or smth?), and the potential itself is modulated by position (eg strikers get 50% penalty on their potential talent for tackling, defenders get 40% on shot precision etc…)