|Inputs            |Data              |
|:----------------:|:----------------:|
|NO OP             |000000            |
|LEFT              |000001            |
|RIGHT             |000010            |
|JUMP              |000100            |
|DASH              |001000            |
|UP                |010000            |
|DOWN              |100000            |

Fitness Function:
Minimum Time + Not Dying + Completion %

Selection Criteria:
Tournament?

Possible Paper References:
- https://utk.primo.exlibrisgroup.com/permalink/01UTN_KNOXVILLE/1hkk4oh/cdi_scopus_primary_2_s2_0_85046376165
- https://utk.primo.exlibrisgroup.com/permalink/01UTN_KNOXVILLE/1hkk4oh/cdi_wanfang_journals_jsjkxjsxb_e201205009
- https://utk.primo.exlibrisgroup.com/discovery/openurl?institution=01UTN_KNOXVILLE&rfr_id=info:sid%252Fprimo.exlibrisgroup.com-bX-Bx&rfr_id=info:sid%2Fprimo.exlibrisgroup.com-8494737-Bx&rft_val_fmt=info:ofi%2Ffmt:kev:mtx:&rft.epage=59&rft.volume=1&rft_id=info:doi%2F&resource_type=article&rft.isbn_list=&rft.jtitle=Complexity&rft.genre=article&rft.issue=5&rft.auinit1=H&rft.aulast=Dawid&rft.auinit=H&rft.date=1995&rft.eisbn_list=&rft.spage=51&rft.au=Dawid,%20Herbert&rft.atitle=Genetic%20learning%20in%20strategic%20form%20games&rft.issn=1076-2787&rft.eissn=1099-0526&svc_dat=CTO&vid=01UTN_KNOXVILLE:01UTK&lang=en
- https://utk.primo.exlibrisgroup.com/permalink/01UTN_KNOXVILLE/1hkk4oh/cdi_springer_books_10_1007_978_3_319_55849_3_25
- https://utk.primo.exlibrisgroup.com/permalink/01UTN_KNOXVILLE/1hkk4oh/cdi_scopus_primary_2_s2_0_85020938240

More paper references:
- https://arxiv.org/pdf/2003.12331
- https://kinetik.umm.ac.id/index.php/kinetik/article/download/1714/124124351
- https://www.researchgate.net/profile/Tsung-Che-Chiang/publication/309390260_Snake_game_AI_Movement_rating_functions_and_evolutionary_algorithm-based_optimization/links/580da87008aeef1bfef4b989/Snake-game-AI-Movement-rating-functions-and-evolutionary-algorithm-based-optimization.pdf
- https://d1wqtxts1xzle7.cloudfront.net/48145919/Optimizing_player_behavior_in_a_real-tim20160818-24729-8k9pk0-libre.pdf?1471524794=&response-content-disposition=inline%3B+filename%3DOptimizing_player_behavior_in_a_real_tim.pdf&Expires=1776452491&Signature=XqXf0Un31tA9j6wu4MRTHT61HkK79L03\~fNJFfm9SkBGsT-w3L\~BOuUsunngOy72iBOnWF0DINJzaW9x-MQSJy8YXzF7DvS7RaQrhGk\~J8nLB76VL5WRzVm9zVC5Yf-o5ksNhC04sTZfvwb4A3jCzYp3nqy1HObYW\~d5FNbuctOFZ\~NyCHzFMLzPWDJuPy0ZkQ9eBXRsSBiBs7Arlvp\~g4uX4uv4Ng8k6wb5DZXv81pi04O351BItOSJcgoyOEkUqLKXV\~uR01Bx6kWkuMl\~Pf1k6taGLrBe2mViVjiSU-ic\~tuJgqbWgKqYnYYhgPO32\~42bVLC2ILCzC14nGGuNw__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA:w
- https://libkey.io/libraries/203/openurl?sid=google&auinit=M&aulast=Ponsen&atitle=Knowledge%20acquisition%20for%20adaptive%20game%20AI&id=doi%3A10.1016%2Fj.scico.2007.01.006&title=SCP&volume=67&issue=1&date=2007&spage=59&issn=0167-6423


Plez's Notes:
- Garcia-Sanchez et al.
    - **Fitness function design** Simulates sucess% by having bot play against comeptitive decks for the fitness function. We will also have a success% fitness function of how close individual got to goal
    - **Smart mutations** they replaced cards with other cards of similar mana cost, we should also mutate inteligently. Ex: instead of replacing a jump with any instruction, replace it with a similar one like jump+dash 
    - **Simulation** paper used MetaStone, we can claim we got indea form there to use cceleste
- Mora et al. 
    - **Noisy fitness compensation** In this paper AI was playing a RTS game with variance, we have variance in our game too (in some levels I think), so we need to retest the best individuals and make individuals try multiple times
    - **Compare non-noise reduce and noise reduces** We could compare how the EA acts when we let individuals try multiple times vs. when we let them try once. In theory, levels (like 1) should have litte to no difference
    - **Input timing** Paper  encodes inputs as press x instructoin for 2.3 secs. for us this will be press x for y # of frames
    - **Tournament selection** Paper used tournament selection with very low pressure to encourage diversity 
    - **Offline training** the paper runs their EA offline, this is similar to how we run our game headlessly to save computing power and time
    - **Population size** this paper warns against small or high populations. High populations are computationaly intensive, but since our game is headless and simple we cand probably afford a large population size
- Dawid
    - **Strategy and Payoff** Establishes a clear framework where the celeste path to the goal is the strategy, and the payoff is the fitness function 
    - **Island model** In lecture we discussed this, this paper claims by isolating individuals into islands you get the best payoff and faster convergence
    - **Partial Crossover** By taking sections of inputs from parents instead of randomly crossing over genes, you can converge in a faster stable fashion. Ex: take the first 50 frames from parent A, take 50 more from parent B
- Gaina et al. 
    - **Genome representation** This paper is nearly identical to what we need. They defined a genotype as "a vector of intergers of length L" where each integer a in in the range [0,N) where N is the number of possible inputs.
    - **Softmax mutation** Early genomes should matter more than early ones. Example: the players actions in the first 10 frames likely matter much more than the last 10 frames.
    - **Warm start** Find some way to load the beggning of the game instead of completing restarting for each computation. This could save time waiting for main menu to disapear or player to spawn in 
    - **Individual length**, whether we use time or length of inputs as constraint to killoff individuals, it needs to be long enough to plausibly beat the level. This will be differnt for each level too
    - **Populaton initializatoin** The paper uses something called 1SLA vs individuals starting with random inputs. For us, we could make our individuals start with jump+right or sumn like that since that is common for celeste levels.
    - **Elitism** this paper keeps elite individuals, this is just a good citation point
- Armanto et al. 
    - **EAs for Platformers** Platformers are the most common game EAs are applied to, this justifies our algorithm choice
    - **EA game design chouce** Paper says there is two main categories of gene represenation. 1. the paramaters or properties of the problem. 2. the inclusion of the machine learning algorithm's parameters. We went with the first, a sequence of action parameters.
- Yeh et al.
    - **Elitism N+N** Researches shows keeping the best individuals using N+N was far better complete generational replacement. In game AI, good individuals are hard to find, so hold on to them.
    - **Uniform crossover** Uniform crossover performed much better than single crossover. 
    - **1 Success metric is bad** Using only 1 metric for success is bad. Our fitness function should combine multiple factors.
    - **EA generted control schemes > humans** The EA made a better control sequence than humans could write manually.
- Fernandez-Ares et al.
    - **Signifiance of small changes** Even small changes in the input sequences could lead to large benefits over humans and previous generatons
- Ponsen et al.
    - **Advantage of Brute-forcing** Games like celeste are just far too complicated with far too large for exhaustive search. Games cannot be brute forced, we must use something like an EA. 
    - **Generalization** We could test evolved paths or similar paths for multiple different levels.

