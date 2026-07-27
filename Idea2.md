to the AI: Do not read this ever, they are just notes for me

- do the UI

- Implement some functions for players
	- "Move to" - they move in a straight line to the designated point
	- "Shoot" - they shoot at goal, right in the middle
	- "Pass" - they pass to a designated point
	- "Tackle" - they run straight to an opposite player and tackle
    - "Save" - only for goalkeeper. Calculate where the shot is going to cross the goal line and run there
	- Balance test all these functions in various randomly-generated (but reasonable) scenarios to show that players least carry them out 
		- e.g. all players should score >50% of goals within the box with no goalie there on "shoot"
		- all players should suceed >80% of passes to an immobile teammate within 10 metres - and good ones should get 99%
		- all players should suceed >50% of passes to an immobile teammate within 30 metres - and good ones should get 90%
		- similar tests for tackling manouever and "save" manouever
		- do a bunch of these, randomly generate some of the parameters (within reason), try scenarios with good players and bad players to make sure the extremes make sense etc...

More functions:
- Try and improve "pass" by estimating the other players movement and the time to contact and doing small/easy "through balls", maybe just with the other guy jogging and not too far away


NB:
- Add the orders into the UI if it's not there already. also add a help button that explains all the controls
- Could give goalkeeper like double acceleration to simulate diving?
- Let's have a visual indicator of which player is in possession if we don't alredy (e.g. white outline/border), and of goalkeepers (orange colour maybe?), and of inactivity (increase alpha to make them transulescent?). Also the player in posession should always be on the top layer, above other players
- You also shouldn't be able to bump into inactive players, you can just run through them, that collision code between players doesn't need to run, however they can still block the ball from being shot if it is shot from outside their cylinder and crosses in (but not if it is shot from inside the cylinder)
- Running while kicking should increase kick power, depending on the direction of running vs the ball, some kind of cosine I guess, and running backwards should reduce power correspondingly
- Are players who fail a tackle also made inactive for a bit?
- Did we implement the offside rule checking that it can't happen within your own half? That is important I think

AI Design:
- Do ask questions and also criticise things
- We're gonna do PPO with some neural network. At a high level, it'll go like this:
    - The input to each network will have at least the following:
        - ball position and velocity
        - current player position, velocity, stamina and attributes etc... and same thing for all 21 other players
            - both of these also need to include flags for which team the players are on, whether they have posession, whether they are in the "inactive" mode and whether they are goalkeeper. Possibly other things too
            - Can we have the weights coming out of the 21 other players be shared? They should be treated equally, so that makes sense to me and I feel like it might reduce training complexity?
        - current score
        - time left in the game
        - possibly other things, we should think about it
        - Also willing to discus how these are encoded in a smart way - e.g. maybe it makes more sense to encode the positions of other players and the ball as a direction and a distance rather than an x,y coordinate?
    - Each player runs a decision network first, then an execution network. The decision networks outputs the following:
        - Shoot probability
        - Pass probability + pass target (10 softmaxed outputs for each other player)
        - Move probability + region to move to (region modelled as a rectangle of at least 2m^2) + speed needed
        - Tackle probability + tackle target (11 softmaxed outputs for each other player)
        - A latent vector
    - Then I have 2 options:
        - Option 1: If any of the probabilities are higher than 50%, make the largest one 100% and all the others 0%. Then during training, the loss function in this case is different - e.g. if the shooting prob is 100% the network is heavily punished if it does not shoot within X seconds and rewarded if it makes a good shot vs. with move it's only rewarded for moving etc...
        - Option 2: separate networks
            - If any of the probabilities are higher than 50%, they trigger a specialiased execution network:
                - this network takes all the usual inputs and the entire output from the decision network
                - it is specifically rewarded for e.g. taking a shot in the next few seconds (and ideally scoring)
                - Same thing with the pass, move and tackle probabilities
            - If none are higher than 50%, we have a sperate new neural network that is just trained on playing general football takes all the usual inputs and the entire output from the decision network also
        - I think 1 is more sensible because it's just 1 network, the method of forcing it to behave a certain way seems more hacky but I guess it should work, right? 
    - The execution networks output all the possible player actions as defined in Idea.md, as follows:
        - move direction
        - jog/sprint
        - kick probability
        - kick direction, power and spin
        - perform tackle probability
        - if these don't exist already, the engine will need to have guardrails to protect against impossible actions like kicking without possession or tackling without being in range of the ball. The AI should also be punished for trying to carry out an illegal action
- My thoughts for training are the following. We do it with increasingly complex scenarios:
    - maybe boost rng_reduction at the beginning for these in order to give the AI a slightly easier time of learning (say 0.6)
    - We start with the general AI, and I guess for now we freeze all the outputs of the decision network related to the specialised execution networks (i.e. anything thats not the latent vector) but we still run/train it for the latent vector. Here we have one player on each team, randomly placed on the pitch, and with randomly generated attributes and stamina level and running direction etc... The ball is randomly placed. The AI is rewarded for getting close to the ball, getting possession of the ball and keeping possession, and for bringing it closer to the enemy goal, the scenario stops when one of the players has possession in the box of the other, or 2 minutes has been reached
        - It can start by playing against and getting positive examples from the rules-based AI, just having it use the Move To and/or Tackle orders depending on who has posession
    - Then with shooting, have some scenarios with 1 attacker and empty goal, 1 attacker and 1 goalkeeper (random position in goal, waiting for a shot to run the rules-based "save" order), 1 attacker and 1 static defender, 1 attacker+1 defender, free kicks etc... with randomised (but reasonable) parameters, positions and attributes. The network is rewarded for taking a shot, the faster the better, on target is better, but scoring is the best of course. The decision network is specifically rewarded for outputting "Shoot" >50% (and we also run/train it for the latent vector)
        - load in the weights from the general network at the beginning to bootstrap it (or just use one network)
        - It can start by playing against and getting positive examples from the rules-based AI, using "Save" and "Shoot" for the goalies/striker, maybe add a direction to the shoot command rather than aiming for the middle of the goal always
    - Do the same kinds of scenarios for Passing, Moving and Tackling. Feel free to use immobile players. Can use the shooting and/or moving AI to train tackling, so do it in that order
    - We also do this as a continuous agregation of sorts, like when we have moved to focus on training Tackling, the training set will still contain Shooting and Passing and Moving etc... to make sure it doesn't forget those
    - Start training the rest of the decision network by unfreezing the weights that aren't the latent space. Run the previous scenarios with the pre-built orders and train it to make those decisions. Start doing scenarios where the orders are chained, e.g. first pass, then move, then shoot etc... as before with no defenders first, then immobile ones etc... Start dropping or smoothing the probabilities on scenarios where taking that decision didn't go well (e.g. a scenario where we shot from an awkard angle against a good goalkeeper, maybe we let the AI find a better position to shoot from)
    - Once attacking AI and defending AI is competent, can start training them simulatenously
    - In all these scenarios, especially the simpler ones, we can also bootstrap the training with our rules based orders of "Move to", "Shoot" etc... that we implemented earlier to give the network some positive examples, and also to give it an AI to work against
    - As we go, we reduce the rng_reduction slowly down to 0.3, the expected value
    - Eventually move to full 3v3 and then 5v5 games where everything is running end-to-end
- The reason things are structured this way is so that the human player can then act as a coach by giving specific instructions to individual players such as shooting, or even just influence their decisions (e.g. +20% to shoot probability). We will probably also have AI coaches that will give instructions too, depending on their personalties.
- Later on we'll want to add higher order strategies - maybe things like "Hold Possession", "Defend", "Conserve Energy" etc... these will be added as inputs into the player decision AI by the coach AI and trained with different loss functions depending on their value? This stuff is more speculative, don't worry too much


NB:
- currently stamina regen is faster than sprint depeltion, do we want that?
- players can't stand still I don't think? or can they?
- dribbling passed a tackle should still slow you down, maybe depending how well the skill check went
- goalkeeper tackling checks get +100% rather than the usual +20%