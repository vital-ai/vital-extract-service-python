from EnriCo.model import EnriCo
import torch, re

from extractservice.utils.config_utils import ConfigUtils

config = ConfigUtils.load_config()

HF_TOKEN = config['api_key_list']['HF_TOKEN']


def process_text_and_relations(model, text, relation_types):
    # Tokenize the text
    def tokenize_text(text):
        return re.findall(r'\w+(?:[-_]\w+)*|\S', text)

    # Remove line breaks from the text and tokenize
    tokens = tokenize_text(text.replace("\n", ""))

    # Prepare the input for the model
    input_x = {"tokenized_text": tokens, "spans": [], "relations": []}

    # Call the model's collate function
    x = model.collate_fn([input_x], relation_types)

    # Predict using the model
    out = model.predict(x, threshold=0.5, output_confidence=True)

    # Process the output to extract and print relations and their confidence
    res = []
    for el in out[0]:
        (s_h, e_h), (s_t, e_t), rtype, conf = el
        head_ = " ".join(tokens[s_h:e_h+1])
        tail_ = " ".join(tokens[s_t:e_t+1])

        if (head_, rtype, tail_) in res:
            continue
        res.append((head_, rtype, tail_))
        print(head_, "---", rtype, "--", tail_, "-- confidence:", conf)


def main():
    print('Hello World')

    model = EnriCo.from_pretrained("urchade/enrico-beta", token=HF_TOKEN)
    model = model.eval()

    text = """
    Mark Elliot Zuckerberg (/ˈzʌkərbɜːrɡ/; born May 14, 1984) is an American businessman and philanthropist. He co-founded the social media service Facebook, along with his Harvard roommates in 2004, and its parent company Meta Platforms (formerly Facebook, Inc.), of which he is chairman, chief executive officer and controlling shareholder.

    Zuckerberg briefly attended Harvard University, where he launched Facebook in February 2004 with his roommates Eduardo Saverin, Andrew McCollum, Dustin Moskovitz and Chris Hughes. Originally launched in only select college campuses, the site expanded rapidly and eventually beyond colleges, reaching one billion users in 2012. Zuckerberg took the company public in May 2012 with majority shares. In 2008, at age 23, he became the world's youngest self-made billionaire. He has since used his funds to organize multiple philanthropic endeavors, including the establishment of the Chan Zuckerberg Initiative.
    """

    relation_types = ["born in", "founder of", "founded in", "studied at"]

    # Parallel prediction
    # print("Parallel prediction:")
    # process_text_and_relations(model, text, relation_types)

    # One-by-one prediction
    # print("\nOne by one prediction (should work better):")
    # for rel in relation_types:
    #    process_text_and_relations(model, text, [rel])

    # text = get_text()

    # text = """John works on the Wilson project."""

    # text = """John works at OpenAI. He started in 2020, but Jane started in 2021."""

    # relation_types = ["employed by", "started in"]

    text = "The net loss of Microsoft in the second quarter of 2022 was $1 billion dollars."

    relation_types = ["loss of", "profit of", "profit in", "loss in"]

    print("\nOne by one prediction (should work better):")

    for rel in relation_types:
        process_text_and_relations(model, text, [rel])


def get_text():

    return """
    OpenAI is an American artificial intelligence (AI) research organization founded in December 2015, researching artificial intelligence with the goal of developing "safe and beneficial" artificial general intelligence, which it defines as "highly autonomous systems that outperform humans at most economically valuable work".
As one of the leading organizations of the AI boom, it has developed several large language models, advanced image generation models, and previously, released open-source models. Its release of ChatGPT has been credited with starting the AI boom.
The organization consists of the non-profit OpenAI, Inc. registered in Delaware and its for-profit subsidiary OpenAI Global, LLC. It was founded by Ilya Sutskever, Greg Brockman, Trevor Blackwell, Vicki Cheung, Andrej Karpathy, Durk Kingma, Jessica Livingston, John Schulman, Pamela Vagata, and Wojciech Zaremba, with Sam Altman and Elon Musk serving as the initial Board of Directors members. Microsoft provided OpenAI Global LLC with a $1 billion investment in 2019 and a $10 billion investment in 2023, with a significant portion of the investment in the form of computational resources on Microsoft's Azure cloud service.
On November 17, 2023, the board removed Altman as CEO, while Brockman was removed as chairman and then resigned as president. Four days later, both returned after negotiations with the board, and most of the board members resigned. The new initial board included former Salesforce co-CEO Bret Taylor as chairman. It was also announced that Microsoft will have a non-voting board seat.

History
2015–2018: Non-profit beginnings
In December 2015, Sam Altman, Greg Brockman, Reid Hoffman, Jessica Livingston, Peter Thiel, Elon Musk, Amazon Web Services (AWS), Infosys, and YC Research announced the formation of OpenAI and pledged over $1 billion to the venture. The actual collected total amount of contributions was only $130 million until 2019. According to an investigation led by TechCrunch, Musk was its largest donor while YC Research did not contribute anything at all. The organization stated it would "freely collaborate" with other institutions and researchers by making its patents and research open to the public. OpenAI is headquartered at the Pioneer Building in Mission District, San Francisco.
According to Wired, Brockman met with Yoshua Bengio, one of the "founding fathers" of deep learning, and drew up a list of the "best researchers in the field". Brockman was able to hire nine of them as the first employees in December 2015. In 2016, OpenAI paid corporate-level (rather than nonprofit-level) salaries, but did not pay AI researchers salaries comparable to those of Facebook or Google.
Microsoft's Peter Lee stated that the cost of a top AI researcher exceeds the cost of a top NFL quarterback prospect. OpenAI's potential and mission drew these researchers to the firm; a Google employee said he was willing to leave Google for OpenAI "partly because of the very strong group of people and, to a very large extent, because of its mission." Brockman stated that "the best thing that I could imagine doing was moving humanity closer to building real AI in a safe way." OpenAI co-founder Wojciech Zaremba stated that he turned down "borderline crazy" offers of two to three times his market value to join OpenAI instead.
In April 2016, OpenAI released a public beta of "OpenAI Gym", its platform for reinforcement learning research. Nvidia gifted its first DGX-1 supercomputer to OpenAI in August 2016 to help it train larger and more complex AI models with the capability of reducing processing time from six days to two hours. In December 2016, OpenAI released "Universe", a software platform for measuring and training an AI's general intelligence across the world's supply of games, websites, and other applications.
In 2017 OpenAI spent $7.9 million, or a quarter of its functional expenses, on cloud computing alone. In comparison, DeepMind's total expenses in 2017 were $442 million. In the summer of 2018, simply training OpenAI's Dota 2 bots required renting 128,000 CPUs and 256 GPUs from Google for multiple weeks.
In 2018, Musk resigned from his Board of Directors seat, citing "a potential future conflict [of interest]" with his role as CEO of Tesla due to Tesla's AI development for self-driving cars. Sam Altman claims that Musk believed OpenAI had fallen behind other players like Google and Musk proposed instead to take over OpenAI himself, which the board rejected. Musk subsequently left OpenAI but claimed to remain a donor, yet made no donations after his departure.
In February 2019, GPT-2 was announced, which gained attention for its ability to generate human-like text.

2019: Transition from non-profit
In 2019, OpenAI transitioned from non-profit to "capped" for-profit, with the profit being capped at 100 times any investment. According to OpenAI, the capped-profit model allows OpenAI Global LLC to legally attract investment from venture funds and, in addition, to grant employees stakes in the company. Many top researchers work for Google Brain, DeepMind, or Facebook, which offer stock options that a nonprofit would be unable to. Before the transition, public disclosure of the compensation of top employees at OpenAI was legally required.
The company then distributed equity to its employees and partnered with Microsoft, announcing an investment package of $1 billion into the company. Since then, OpenAI systems have run on an Azure-based supercomputing platform from Microsoft.
OpenAI Global LLC then announced its intention to commercially license its technologies. It planned to spend the $1 billion "within five years, and possibly much faster." Altman has stated that even a billion dollars may turn out to be insufficient, and that the lab may ultimately need "more capital than any non-profit has ever raised" to achieve artificial general intelligence.
The transition from a nonprofit to a capped-profit company was viewed with skepticism by Oren Etzioni of the nonprofit Allen Institute for AI, who agreed that wooing top researchers to a nonprofit is difficult, but stated "I disagree with the notion that a nonprofit can't compete" and pointed to successful low-budget projects by OpenAI and others. "If bigger and better funded was always better, then IBM would still be number one."
The nonprofit, OpenAI, Inc., is the sole controlling shareholder of OpenAI Global LLC, which, despite being a for-profit company, retains a formal fiduciary responsibility to OpenAI, Inc.'s nonprofit charter. A majority of OpenAI, Inc.'s board is barred from having financial stakes in OpenAI Global LLC. In addition, minority members with a stake in OpenAI Global LLC are barred from certain votes due to conflict of interest. Some researchers have argued that OpenAI Global LLC's switch to for-profit status is inconsistent with OpenAI's claims to be "democratizing" AI.

2020–2023: ChatGPT, DALL-E, partnership with Microsoft
In 2020, OpenAI announced GPT-3, a language model trained on large internet datasets. GPT-3 is aimed at natural language answering questions, but it can also translate between languages and coherently generate improvised text. It also announced that an associated API, named simply "the API", would form the heart of its first commercial product.
In 2021, OpenAI introduced DALL-E, a specialized deep learning model adept at generating complex digital images from textual descriptions, utilizing a variant of the GPT-3 architecture.
In December 2022, OpenAI received widespread media coverage after launching a free preview of ChatGPT, its new AI chatbot based on GPT-3.5. According to OpenAI, the preview received over a million signups within the first five days. According to anonymous sources cited by Reuters in December 2022, OpenAI Global LLC was projecting $200 million of revenue in 2023 and $1 billion in revenue in 2024.
In January 2023, OpenAI Global LLC was in talks for funding that would value the company at $29 billion, double its 2021 value. On January 23, 2023, Microsoft announced a new US$10 billion investment in OpenAI Global LLC over multiple years, partially needed to use Microsoft's cloud-computing service Azure. Rumors of this deal suggested that Microsoft may receive 75% of OpenAI's profits until it secures its investment return and a 49% stake in the company. The investment is believed to be a part of Microsoft's efforts to integrate OpenAI's ChatGPT into the Bing search engine. Google announced a similar AI application (Bard), after ChatGPT was launched, fearing that ChatGPT could threaten Google's place as a go-to source for information.
On February 7, 2023, Microsoft announced that it was building AI technology based on the same foundation as ChatGPT into Microsoft Bing, Edge, Microsoft 365 and other products.
On March 3, 2023, Reid Hoffman resigned from his board seat, citing a desire to avoid conflicts of interest with his investments in AI companies via Greylock Partners, and his co-founding of the AI startup Inflection AI. Hoffman remained on the board of Microsoft, a major investor in OpenAI.
On March 14, 2023, OpenAI released GPT-4, both as an API (with a waitlist) and as a feature of ChatGPT Plus.
On May 22, 2023, Sam Altman, Greg Brockman and Ilya Sutskever posted recommendations for the governance of superintelligence. They consider that superintelligence could happen within the next 10 years, allowing a "dramatically more prosperous future" and that "given the possibility of existential risk, we can't just be reactive". They propose creating an international watchdog organization similar to IAEA to oversee AI systems above a certain capability threshold, suggesting that relatively weak AI systems on the other side should not be overly regulated. They also call for more technical safety research for superintelligences, and ask for more coordination, for example through governments launching a joint project which "many current efforts become part of".
In August 2023, it was announced that OpenAI had acquired the New York-based start-up, Global Illumination, a company that deploys AI to develop digital infrastructure and creative tools.
In October 2023, Sam Altman and Peng Xiao, CEO of the Emirati AI firm G42, announced Open AI would let G42 deploy Open AI technology.
On November 6, 2023, OpenAI launched GPTs, allowing individuals to create customized versions of ChatGPT for specific purposes, further expanding the possibilities of AI applications across various industries. On November 14, 2023, OpenAI announced they temporarily suspended new sign-ups for ChatGPT Plus due to high demand. Access for newer subscribers re-opened a month later on December 13.
On September 21, 2023, Microsoft had begun rebranding all variants of its Copilot to Microsoft Copilot, including the former Bing Chat and the Microsoft 365 Copilot. This strategy was followed in December 2023 by adding the MS-Copilot to many installations of Windows 11 and Windows 10 as well as a standalone Microsoft Copilot app released for Android and one released for iOS thereafter.

Brief departure of Altman and Brockman
On November 17, 2023, Sam Altman was removed as CEO when its board of directors (composed of Helen Toner, Ilya Sutskever, Adam D'Angelo and Tasha McCauley) cited a lack of confidence in him. Chief Technology Officer Mira Murati took over as interim CEO. Greg Brockman, the president of OpenAI, was also removed as chairman of the board and resigned from the company's presidency shortly thereafter. Three senior OpenAI researchers subsequently resigned: director of research and GPT-4 lead Jakub Pachocki, head of AI risk Aleksander Madry, and researcher Szymon Sidor.
On November 18, 2023, there were reportedly talks of Altman returning as CEO amid pressure placed upon the board by investors such as Microsoft and Thrive Capital, who objected to Altman's departure. Although Altman himself spoke in favor of returning to OpenAI, he has since stated that he considered starting a new company and bringing former OpenAI employees with him if talks to reinstate him didn't work out. The board members agreed "in principle" to resign if Altman returned. On November 19, 2023, negotiations with Altman to return failed and Murati was replaced by Emmett Shear as interim CEO. The board initially contacted Anthropic CEO Dario Amodei (a former OpenAI executive) about replacing Altman, and proposed a merger of the two companies, but both offers were declined.
On November 20, 2023, Microsoft CEO Satya Nadella announced Altman and Brockman would be joining Microsoft to lead a new advanced AI research team, but added that they were still committed to OpenAI despite recent events. Before the partnership with Microsoft was finalized, Altman gave the board another opportunity to negotiate with him. About 738 of OpenAI's 770 employees, including Murati and Sutskever, signed an open letter stating they would quit their jobs and join Microsoft if the board did not rehire Altman and then resign. This prompted OpenAI investors to consider legal action against the board as well. In response, OpenAI management sent an internal memo to employees stating that negotiations with Altman and the board had resumed and would take some time.

On November 21, 2023, after continued negotiations, Altman and Brockman returned to the company in their prior roles along with a reconstructed board made up of new members Bret Taylor (as chairman) and Lawrence Summers, with D'Angelo remaining. On November 22, 2023, emerging reports suggested that Sam Altman's dismissal from OpenAI may have been linked to his alleged mishandling of a significant breakthrough in the organization's secretive project codenamed Q*. According to sources within OpenAI, Q* is aimed at developing AI capabilities in logical and mathematical reasoning, and reportedly involves performing math on the level of grade-school students. Concerns about Altman's response to this development, specifically regarding the discovery's potential safety implications, were reportedly raised with the company's board shortly before Altman's firing. On November 29, 2023, OpenAI announced that an anonymous Microsoft employee had joined the board as a non-voting member to observe the company's operations.
In February 2024, the U.S. Securities and Exchange Commission was reportedly investigating OpenAI over whether internal company communications made by Altman were used to mislead investors; and an investigation of Altman's statements, opened by the Southern New York U.S. Attorney's Office the previous November, was ongoing.

2024–present: Public/non-profit efforts, Sora
On January 16, 2024, in response to intense scrutiny from regulators around the world, OpenAI announced the formation of a new Collective Alignment team that would aim to implement ideas from the public for ensuring its models would "align to the values of humanity." The move was from its public program launched in May 2023. The company explained that the program would be separate from its commercial endeavors. On January 18, 2024, OpenAI announced a partnership with Arizona State University that would give it complete access to ChatGPT Enterprise. ASU plans to incorporate the technology into various aspects of its operations, including courses, tutoring and research. It is OpenAI's first partnership with an educational institution.
On February 15, 2024, OpenAI announced a text-to-video model named Sora, which it plans to release to the public at an unspecified date. It is currently available for red teams for managing critical harms and risks.
On February 29, 2024, OpenAI and CEO Sam Altman were sued by Elon Musk, who accused them of prioritizing profits over public good, contrary to OpenAI's original mission of developing AI for humanity's benefit. The lawsuit cited OpenAI's policy shift after partnering with Microsoft, questioning its open-source commitment and stirring the AI ethics-vs.-profit debate. In a blog post, OpenAI stated that "Elon understood the mission did not imply open-sourcing AGI." In a staff memo, they also denied being a de facto Microsoft subsidiary.
In a March 11, 2024, court filing, OpenAI said it was "doing just fine without Elon Musk" after he left the company in 2018. They also responded to Musk's lawsuit, calling the billionaire’s claims "incoherent", "frivolous", "extraordinary" and "a fiction".
On May 13, 2024, OpenAI introduced a new AI model called GPT-4o. The "o" stands for "omni", which means it can handle text, speech, and video. Over the next few weeks, GPT-4o will be gradually introduced across OpenAI's products for developers and consumers. On May 15, Ilya Sutskever resigned from OpenAI and was replaced with Jakub Pachocki to be the Chief Scientist.

Participants
Key employees

CEO and co-founder: Sam Altman, former president of the startup accelerator Y Combinator
President and co-founder: Greg Brockman, former CTO, 3rd employee of Stripe
Chief Scientist: Jakub Pachocki, former Director of Research at OpenAI
Chief Technology Officer: Mira Murati, previously at Leap Motion and Tesla, Inc.
Chief Operating Officer: Brad Lightcap, previously at Y Combinator and JPMorgan Chase

Board of Directors of the OpenAI nonprofit
Microsoft (observer)
Bret Taylor (chairman)
Sam Altman
Lawrence Summers
Adam D'Angelo
Sue Desmond-Hellmann
Nicole Seligman
Fidji Simo

Principal individual investors
Reid Hoffman, LinkedIn co-founder
Peter Thiel, PayPal co-founder
Jessica Livingston, a founding partner of Y Combinator
Elon Musk, co-founder

Corporate investors
Microsoft
Khosla Ventures
Infosys
Thrive Capital

Motives
Some scientists, such as Stephen Hawking and Stuart Russell, have articulated concerns that if advanced AI gains the ability to redesign itself at an ever-increasing rate, an unstoppable "intelligence explosion" could lead to human extinction. Co-founder Musk characterizes AI as humanity's "biggest existential threat".
Musk and Altman have stated they are partly motivated by concerns about AI safety and the existential risk from artificial general intelligence. OpenAI states that "it's hard to fathom how much human-level AI could benefit society," and that it is equally difficult to comprehend "how much it could damage society if built or used incorrectly". Research on safety cannot safely be postponed: "because of AI's surprising history, it's hard to predict when human-level AI might come within reach." OpenAI states that AI "should be an extension of individual human wills and, in the spirit of liberty, as broadly and evenly distributed as possible." Co-chair Sam Altman expects the decades-long project to surpass human intelligence.
Vishal Sikka, former CEO of Infosys, stated that an "openness", where the endeavor would "produce results generally in the greater interest of humanity", was a fundamental requirement for his support; and that OpenAI "aligns very nicely with our long-held values" and their "endeavor to do purposeful work". Cade Metz of Wired suggested that corporations such as Amazon might be motivated by a desire to use open-source software and data to level the playing field against corporations such as Google and Facebook, which own enormous supplies of proprietary data. Altman stated that Y Combinator companies would share their data with OpenAI.

Strategy
In the early years before his 2018 departure, Musk posed the question: "What is the best thing we can do to ensure the future is good? We could sit on the sidelines or we can encourage regulatory oversight, or we could participate with the right structure with people who care deeply about developing AI in a way that is safe and is beneficial to humanity." He acknowledged that "there is always some risk that in actually trying to advance (friendly) AI we may create the thing we are concerned about"; but nonetheless, that the best defense was "to empower as many people as possible to have AI. If everyone has AI powers, then there's not any one person or a small set of individuals who can have AI superpower."
Musk and Altman's counterintuitive strategy—that of trying to reduce of harm from AI by giving everyone access to it—is controversial among those concerned with existential risk from AI. Philosopher Nick Bostrom said, "If you have a button that could do bad things to the world, you don't want to give it to everyone." During a 2016 conversation about technological singularity, Altman said, "We don't plan to release all of our source code" and mentioned a plan to "allow wide swaths of the world to elect representatives to a new governance board". Greg Brockman stated, "Our goal right now... is to do the best thing there is to do. It's a little vague."
Conversely, OpenAI's initial decision to withhold GPT-2 around 2019, due to a wish to "err on the side of caution" in the presence of potential misuse, was criticized by advocates of openness. Delip Rao, an expert in text generation, stated, "I don't think [OpenAI] spent enough time proving [GPT-2] was actually dangerous." Other critics argued that open publication was necessary to replicate the research and to create countermeasures.
More recently, in 2022, OpenAI published its approach to the alignment problem, anticipating that aligning AGI to human values would likely be harder than aligning current AI systems: "Unaligned AGI could pose substantial risks to humanity[,] and solving the AGI alignment problem could be so difficult that it will require all of humanity to work together". They stated that they intended to explore how to better use human feedback to train AI systems, and how to safely use AI to incrementally automate alignment research. Some observers believe the company's November 2023 reorganization—including Altman's return as CEO, and the changes to its board of directors—indicated a probable shift towards a business focus and reduced influence of "cautious people" at OpenAI.

Products and applications
Reinforcement learning
At its beginning, OpenAI's research included many projects focused on reinforcement learning (RL). OpenAI has been viewed as an important competitor to DeepMind.

Gym
Announced in 2016, Gym aimed to provide an easily implemented general-intelligence benchmark in a wide variety of environments—akin to, but broader than, the ImageNet Large Scale Visual Recognition Challenge used in supervised learning research. It sought to standardize how environments were defined in AI research publications, so that published research became more easily reproducible, and to provide users with a simple interface. As of June 2017, Gym could be used only with Python. As of September 2017, the Gym documentation site was not maintained, and active work focused instead on its GitHub page.

Gym Retro
Released in 2018, Gym Retro is a platform for reinforcement learning (RL) research on video games, using RL algorithms and study generalization. Prior RL research focused mainly on optimizing agents to solve single tasks. Gym Retro gives the ability to generalize between games with similar concepts but different appearances.

RoboSumo
Released in 2017, RoboSumo is a virtual world where humanoid metalearning robot agents initially lack knowledge of how to even walk, but are given the goals of learning to move and to push the opposing agent out of the ring. Through this adversarial learning process, the agents learn how to adapt to changing conditions. When an agent is then removed from this virtual environment and placed in a new virtual environment with high winds, the agent braces to remain upright, suggesting it had learned how to balance in a generalized way. OpenAI's Igor Mordatch argued that competition between agents could create an intelligence "arms race" that could increase an agent's ability to function even outside the context of the competition.

OpenAI Five
OpenAI Five is a team of five OpenAI-curated bots used in the competitive five-on-five video game Dota 2, that learn to play against human players at a high skill level entirely through trial-and-error algorithms. Before becoming a team of five, the first public demonstration occurred at The International 2017, the annual premiere championship tournament for the game, where Dendi, a professional Ukrainian player, lost against a bot in a live one-on-one matchup. After the match, CTO Greg Brockman explained that the bot had learned by playing against itself for two weeks of real time, and that the learning software was a step in the direction of creating software that can handle complex tasks like a surgeon. The system uses a form of reinforcement learning, as the bots learn over time by playing against themselves hundreds of times a day for months, and are rewarded for actions such as killing an enemy and taking map objectives.
By June 2018, the ability of the bots expanded to play together as a full team of five, and they were able to defeat teams of amateur and semi-professional players. At The International 2018, OpenAI Five played in two exhibition matches against professional players, but ended up losing both games. In April 2019, OpenAI Five defeated OG, the reigning world champions of the game at the time, 2:0 in a live exhibition match in San Francisco. The bots' final public appearance came later that month, where they played in 42,729 total games in a four-day open online competition, winning 99.4% of those games.
OpenAI Five's mechanisms in Dota 2's bot player shows the challenges of AI systems in multiplayer online battle arena (MOBA) games and how OpenAI Five has demonstrated the use of deep reinforcement learning (DRL) agents to achieve superhuman competence in Dota 2 matches.

Dactyl
Developed in 2018, Dactyl uses machine learning to train a Shadow Hand, a human-like robot hand, to manipulate physical objects. It learns entirely in simulation using the same RL algorithms and training code as OpenAI Five. OpenAI tackled the object orientation problem by using domain randomization, a simulation approach which exposes the learner to a variety of experiences rather than trying to fit to reality. The set-up for Dactyl, aside from having motion tracking cameras, also has RGB cameras to allow the robot to manipulate an arbitrary object by seeing it. In 2018, OpenAI showed that the system was able to manipulate a cube and an octagonal prism.
In 2019, OpenAI demonstrated that Dactyl could solve a Rubik's Cube. The robot was able to solve the puzzle 60% of the time. Objects like the Rubik's Cube introduce complex physics that is harder to model. OpenAI did this by improving the robustness of Dactyl to perturbations by using Automatic Domain Randomization (ADR), a simulation approach of generating progressively more difficult environments. ADR differs from manual domain randomization by not needing a human to specify randomization ranges.

API
In June 2020, OpenAI announced a multi-purpose API which it said was "for accessing new AI models developed by OpenAI" to let developers call on it for "any English language AI task".

Text generation
The company has popularized generative pretrained transformers (GPT).

OpenAI's original GPT model ("GPT-1")
The original paper on generative pre-training of a transformer-based language model was written by Alec Radford and his colleagues, and published in preprint on OpenAI's website on June 11, 2018. It showed how a generative model of language could acquire world knowledge and process long-range dependencies by pre-training on a diverse corpus with long stretches of contiguous text.

GPT-2
Generative Pre-trained Transformer 2 ("GPT-2") is an unsupervised transformer language model and the successor to OpenAI's original GPT model ("GPT-1"). GPT-2 was announced in February 2019, with only limited demonstrative versions initially released to the public. The full version of GPT-2 was not immediately released due to concern about potential misuse, including applications for writing fake news. Some experts expressed skepticism that GPT-2 posed a significant threat.
In response to GPT-2, the Allen Institute for Artificial Intelligence responded with a tool to detect "neural fake news". Other researchers, such as Jeremy Howard, warned of "the technology to totally fill Twitter, email, and the web up with reasonable-sounding, context-appropriate prose, which would drown out all other speech and be impossible to filter". In November 2019, OpenAI released the complete version of the GPT-2 language model. Several websites host interactive demonstrations of different instances of GPT-2 and other transformer models.
GPT-2's authors argue unsupervised language models to be general-purpose learners, illustrated by GPT-2 achieving state-of-the-art accuracy and perplexity on 7 of 8 zero-shot tasks (i.e. the model was not further trained on any task-specific input-output examples).
The corpus it was trained on, called WebText, contains slightly 40 gigabytes of text from URLs shared in Reddit submissions with at least 3 upvotes. It avoids certain issues encoding vocabulary with word tokens by using byte pair encoding. This permits representing any string of characters by encoding both individual characters and multiple-character tokens.

GPT-3
First described in May 2020, Generative Pre-trained Transformer 3 (GPT-3) is an unsupervised transformer language model and the successor to GPT-2. OpenAI stated that the full version of GPT-3 contained 175 billion parameters, two orders of magnitude larger than the 1.5 billion in the full version of GPT-2 (although GPT-3 models with as few as 125 million parameters were also trained).
OpenAI stated that GPT-3 succeeded at certain "meta-learning" tasks and could generalize the purpose of a single input-output pair. The GPT-3 release paper gave examples of translation and cross-linguistic transfer learning between English and Romanian, and between English and German.
GPT-3 dramatically improved benchmark results  over GPT-2. OpenAI cautioned that such scaling-up of language models could be approaching or encountering the fundamental capability limitations of predictive language models. Pre-training GPT-3 required several thousand petaflop/s-days of compute, compared to tens of petaflop/s-days for the full GPT-2 model. Like its predecessor, the GPT-3 trained model was not immediately released to the public for concerns of possible abuse, although OpenAI planned to allow access through a paid cloud API after a two-month free private beta that began in June 2020.
On September 23, 2020, GPT-3 was licensed exclusively to Microsoft.

Codex
Announced in mid-2021, Codex is a descendant of GPT-3 that has additionally been trained on code from 54 million GitHub repositories, and is the AI powering the code autocompletion tool GitHub Copilot. In August 2021, an API was released in private beta. According to OpenAI, the model can create working code in over a dozen programming languages, most effectively in Python.
Several issues with glitches, design flaws and security vulnerabilities were cited.
GitHub Copilot has been accused of emitting copyrighted code, with no author attribution or license.
OpenAI announced that they would discontinue support for Codex API on March 23, 2023.

GPT-4
On March 14, 2023, OpenAI announced the release of Generative Pre-trained Transformer 4 (GPT-4), capable of accepting text or image inputs. They announced that the updated technology passed a simulated law school bar exam with a score around the top 10% of test takers. (By contrast, GPT-3.5 scored around the bottom 10%.) They said that GPT-4 could also read, analyze or generate up to 25,000 words of text, and write code in all major programming languages.
Observers reported that the iteration of ChatGPT using GPT-4 was an improvement on the previous GPT-3.5-based iteration, with the caveat that GPT-4 retained some of the problems with earlier revisions. GPT-4 is also capable of taking images as input on ChatGPT. OpenAI has declined to reveal various technical details and statistics about GPT-4, such as the precise size of the model.

Image classification
CLIP
Revealed in 2021, CLIP (Contrastive Language–Image Pre-training) is a model that is trained to analyze the semantic similarity between text and images. It can notably be used for image classification.

Text-to-image
DALL-E
Revealed in 2021, DALL-E is a Transformer model that creates images from textual descriptions. DALL-E uses a 12-billion-parameter version of GPT-3 to interpret natural language inputs (such as "a green leather purse shaped like a pentagon" or "an isometric view of a sad capybara") and generate corresponding images. It can create images of realistic objects ("a stained-glass window with an image of a blue strawberry") as well as objects that do not exist in reality ("a cube with the texture of a porcupine"). As of March 2021, no API or code is available.

DALL-E 2
In April 2022, OpenAI announced DALL-E 2, an updated version of the model with more realistic results. In December 2022, OpenAI published on GitHub software for Point-E, a new rudimentary system for converting a text description into a 3-dimensional model.

DALL-E 3
In September 2023, OpenAI announced DALL-E 3, a more powerful model better able to generate images from complex descriptions without manual prompt engineering and render complex details like hands and text. It was released to the public as a ChatGPT Plus feature in October.

Text-to-video
Sora
Sora is a text-to-video model that can generate videos based on short descriptive prompts as well as extend existing videos forwards or backwards in time. It can generate videos with resolution up to 1920x1080 or 1080x1920. The maximal length of generated videos is unknown.
Sora's development team named it after the Japanese word for "sky", to signify its "limitless creative potential". Sora's technology is an adaptation of the technology behind the DALL·E 3 text-to-image model. OpenAI trained the system using publicly-available videos as well as copyrighted videos licensed for that purpose, but did not reveal the number or the exact sources of the videos.
OpenAI demonstrated some Sora-created high-definition videos to the public on February 15, 2024, stating that it could generate videos up to one minute long. It also shared a technical report highlighting the methods used to train the model, and the model's capabilities. It acknowledged some of its shortcomings, including struggles simulating complex physics. Will Douglas Heaven of the MIT Technology Review called the demonstration videos "impressive", but noted that they must have been cherry-picked and might not represent Sora's typical output.
Despite skepticism from some academic leaders following Sora's public demo, notable entertainment-industry figures have shown significant interest in the technology's potential. In an interview, actor/filmmaker Tyler Perry expressed his astonishment at the technology's ability to generate realistic video from text descriptions, citing its potential to revolutionize storytelling and content creation. He said that his excitement about Sora's possibilities was so strong that he had decided to pause plans for expanding his Atlanta-based movie studio.

Speech-to-text
Whisper
Released in 2022, Whisper is a general-purpose speech recognition model. It is trained on a large dataset of diverse audio and is also a multi-task model that can perform multilingual speech recognition as well as speech translation and language identification.

Music generation
MuseNet
Released in 2019, MuseNet is a deep neural net trained to predict subsequent musical notes in MIDI music files. It can generate songs with 10 instruments in 15 styles. According to The Verge, a song generated by MuseNet tends to start reasonably but then fall into chaos the longer it plays. In pop culture, initial applications of this tool were used as early as 2020 for the internet psychological thriller Ben Drowned to create music for the titular character.

Jukebox
Released in 2020, Jukebox is an open-sourced algorithm to generate music with vocals. After training on 1.2 million samples, the system accepts a genre, artist, and a snippet of lyrics and outputs song samples. OpenAI stated the songs "show local musical coherence [and] follow traditional chord patterns" but acknowledged that the songs lack "familiar larger musical structures such as choruses that repeat" and that "there is a significant gap" between Jukebox and human-generated music. The Verge stated "It's technologically impressive, even if the results sound like mushy versions of songs that might feel familiar", while Business Insider stated "surprisingly, some of the resulting songs are catchy and sound legitimate".

User interfaces
Debate Game
In 2018, OpenAI launched the Debate Game, which teaches machines to debate toy problems in front of a human judge. The purpose is to research whether such an approach may assist in auditing AI decisions and in developing explainable AI.

Microscope
Released in 2020, Microscope is a collection of visualizations of every significant layer and neuron of eight neural network models which are often studied in interpretability. Microscope was created to analyze the features that form inside these neural networks easily. The models included are AlexNet, VGG 19, different versions of Inception, and different versions of CLIP Resnet.

ChatGPT
Launched in November 2022, ChatGPT is an artificial intelligence tool built on top of GPT-3 that provides a conversational interface that allows users to ask questions in natural language. The system then responds with an answer within seconds. ChatGPT reached 1 million users 5 days after its launch.
As of 2023, ChatGPT Plus is a GPT-4 backed version of ChatGPT available for a US$20 per month subscription fee (the original version is backed by GPT-3.5). OpenAI also makes GPT-4 available to a select group of applicants through their GPT-4 API waitlist; after being accepted, an additional fee of US$0.03 per 1000 tokens in the initial text provided to the model ("prompt"), and US$0.06 per 1000 tokens that the model generates ("completion"), is charged for access to the version of the model with an 8192-token context window; for the 32768-token context window, the prices are doubled.
In May 2023, OpenAI launched a user interface for ChatGPT for the App Store on iOS and later in July 2023 for the Play Store on Android. The app supports chat history syncing and voice input (using Whisper, OpenAI's speech recognition model). In September 2023, OpenAI announced that ChatGPT "can now see, hear, and speak". ChatGPT Plus users can upload images, while mobile app users can talk to the chatbot.
In October 2023, OpenAI's latest image generation model, DALL-E 3, was integrated into ChatGPT Plus and ChatGPT Enterprise. The integration uses ChatGPT to write prompts for DALL-E guided by conversation with users.
OpenAI's GPT Store, initially slated for a 2023 launch, is now deferred to an undisclosed date in early 2024, attributed likely to the leadership changes in November following the initial announcement.

Controversies
In January 2023, OpenAI has been criticized for outsourcing the annotation of data sets to Sama, a company based in San Francisco but employing workers in Kenya. These annotations were used to train an AI model to detect toxicity, which could then be used to filter out toxic content, notably from ChatGPT's training data and outputs. However, these pieces of text usually contained detailed descriptions of various types of violence, including sexual violence. The four Sama employees interviewed by Time described themselves as mentally scarred. OpenAI paid Sama $12.50 per hour of work, and Sama was redistributing the equivalent of between $1.32 and $2.00 per hour post-tax to its annotators. Sama's spokesperson said that the $12.50 was also covering other implicit costs, among which were infrastructure expenses, quality assurance and management.
In March 2023, the company was also criticized for disclosing particularly few technical details about products like GPT-4, contradicting its initial commitment to openness and making it harder for independent researchers to replicate its work and develop safeguards. OpenAI cited competitiveness and safety concerns to justify this strategic turn. OpenAI's chief scientist Ilya Sutskever argued in 2023 that open-sourcing increasingly capable models was increasingly risky, and that the safety reasons for not open-sourcing the most potent AI models would become "obvious" in a few years.
OpenAI was sued for copyright infringement by authors Sarah Silverman, Matthew Butterick, Paul Tremblay and Mona Awad in July 2023. In September 2023, 17 authors, including George R. R. Martin, John Grisham, Jodi Picoult and Jonathan Franzen, joined the Authors Guild in filing a class action lawsuit against OpenAI, alleging that the company's technology was illegally using their copyrighted work. The New York Times also sued the company in late December 2023. In May 2024 it was revealed that OpenAI had destroyed its Books1 and Books2 training datasets, which were used in the training of GPT-3, and which the Authors Guild believed to have contained over 100,000 copyrighted books.
OpenAI was sued for violating EU General Data Protection Regulations in August 2023. In April 2023, the EU's European Data Protection Board (EDPB) formed a dedicated task force on ChatGPT "to foster cooperation and to exchange information on possible enforcement actions conducted by data protection authorities" based on the "enforcement action undertaken by the Italian data protection authority against Open AI about the Chat GPT service".
On February 2024, The Intercept as well as Raw Story and Alternate Media Inc. filed lawsuit against OpenAI on copyright litigation ground. The lawsuit is said to have charted a new legal strategy for digital-only publishers to sue OpenAI.
In late April 2024 NOYB filed a complaint with the Austrian Datenschutzbehörde against OpenAI for violating the European General Data Protection Regulation (GDPR). A text, created with the ChatGPT, gave a false date of birth for a living person without giving the individual the option to see the personal data used in the process. A request to correct the mistake was denied. Additionally, neither the recipients of ChatGPT´s work nor the sources used, could be made available, OpenAI claimed.
On April 30, 2024, eight newspapers filed a lawsuit in the Southern District of New York against OpenAI and Microsoft, claiming illegal harvesting of their copyrighted articles. The suing publications included The Mercury News, The Denver Post, The Orange County Register, St. Paul Pioneer-Press, Chicago Tribune, Orlando Sentinel, South Florida Sun Sentinel, and New York Daily News.

Removal of military and warfare clause
OpenAI quietly deleted its ban on using ChatGPT for "military and warfare". Up until January 10, 2024, its "usage policies" included a ban on "activity that has high risk of physical harm, including," specifically, "weapons development" and "military and warfare." Its new policies prohibit "[using] our service to harm yourself or others" and to "develop or use weapons". As one of the industry collaborators, OpenAI provides LLM to the Artificial Intelligence Cyber Challenge (AIxCC) sponsored by Defense Advanced Research Projects Agency and Advanced Research Projects Agency for Health to protect software critical to Americans.

See also
Anthropic
Center for AI Safety
Future of Humanity Institute
Future of Life Institute
Google DeepMind
Machine Intelligence Research Institute
Microsoft
Bing
The New York Times

Notes
References
External links

Official website 
OpenAI on Twitter
OpenAI on Instagram
OpenAI on YouTube
"What OpenAI Really Wants" by Wired
"The Inside Story of Microsoft's Partnership with OpenAI" by The New Yorker    
    """


if __name__ == "__main__":
    main()





