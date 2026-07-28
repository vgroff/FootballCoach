to the AI: Do not read this ever, they are just notes for me

Read Idea.md, and all the READMe and knowledge files, except Idea2
We have implemented what is in Idea.md. the following remainds:

More functions:
- Try and improve "pass" by estimating the other players movement and the time to contact and doing small/easy "through balls", maybe just with the other guy jogging and not too far away
- fleshing out tackling - angle of the tackle should give bonuses and especially maluses when e.g. tackling from behind (i'm thinking add another modifier to the tackling roll, +10% for a fully frontal tackle, down to -5% for a 90 degree tackle, -65% for a fully 180 degree tackle)
- when running at the ball during "get possession", try to run ahead of it (or the player) by estimating where it will be using the velocity and yours (do a rough approximation)
- This is no pass action! Only kicks. I'd rather understand why kick isn't working well as a pass than fake something like that

New NBs:
-     kicker = make_player(
        "k", Team.LEFT, position=Vector3(x, y, 0),
        kick_precision=precision, kick_power=power,
    )
    ball = Ball.at_rest(kicker.position)
    ball.possessed_by = "k"
- this is bad code, no? why wouldn't you do ball.possesed_by = kicker? What if two players have the same name? have a look in the codebase for other places where references are made with string literals instead of with objects. This is the only one I could find at least



NB:
- currently stamina regen is faster than sprint depeltion, do we want that?
- Pretty sure you can currently kick at full power with having full in the kick_power stat, which begs the question of what it's purpose is. I think you should only have full power at 1.0 kick power, but maybe we up the max power a little bit too since it is very rare. I want the kicking/shooting to be realistic
- In the move to order, avoid nearby players (a kind of repulsion mechanic with distance)
