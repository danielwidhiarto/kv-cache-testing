"""AES Dataset Loader — Utilities for loading ASAP 2.0 essay data and formatting long-context LLM prompts."""

import os
import shutil
import pandas as pd
from typing import List, Dict, Any, Optional


class AESDatasetLoader:
    """Loads student essays, rubrics, and source texts from ASAP 2.0 CSV dataset."""

    def __init__(self, csv_path: str = "dataset/ASAP2_train_sourcetexts.csv"):
        zip_path = "dataset/ASAP2_train_sourcetexts.zip"
        if os.path.exists(zip_path) and not self._is_asap_csv(csv_path):
            self._extract_asap_csv(zip_path, csv_path)

        if not os.path.exists(csv_path):
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            self._generate_fallback_dataset(csv_path)
        self.csv_path = csv_path
        self._df: Optional[pd.DataFrame] = None

    @staticmethod
    def _is_asap_csv(csv_path: str) -> bool:
        """Return whether a CSV has the real ASAP 2.0 schema.

        The repository archive stores the CSV with a ``dataset/`` prefix.
        Extracting it into ``dataset/`` naively creates ``dataset/dataset`` and
        makes the loader incorrectly fall back to synthetic data.
        """
        if not os.path.exists(csv_path):
            return False
        try:
            columns = set(pd.read_csv(csv_path, nrows=0).columns)
            return {"essay_id", "score", "full_text", "assignment", "source_text_1"}.issubset(columns)
        except Exception:
            return False

    @staticmethod
    def _extract_asap_csv(zip_path: str, csv_path: str) -> None:
        import zipfile

        print(f"📦 Extracting ASAP 2.0 dataset from {zip_path}...")
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            candidates = [
                name for name in zip_ref.namelist()
                if name.rstrip("/").endswith(os.path.basename(csv_path))
            ]
            if not candidates:
                raise FileNotFoundError(
                    f"Could not find {os.path.basename(csv_path)} inside {zip_path}"
                )
            with zip_ref.open(candidates[0], "r") as source, open(csv_path, "wb") as target:
                shutil.copyfileobj(source, target)
        print(f"✓ ASAP 2.0 dataset extracted successfully to {csv_path}!")


    def _generate_fallback_dataset(self, csv_path: str):
        """Generates realistic ASAP 2.0 dataset samples with LONG essays for Google Colab if local file is missing."""
        print(f"⚠️ Dataset file missing at {csv_path}. Creating fallback ASAP 2.0 long-context samples...")

        # Long source texts + assignments to produce 2000-3500 token prompts
        topics = [
            (
                "Exploring Venus",
                """Venus is the second planet from the Sun and is often called Earth's twin due to its similar size and composition.
However, the resemblance ends there. Venus has a thick, toxic atmosphere composed mainly of carbon dioxide, with clouds of sulfuric acid.
Surface temperatures on Venus average around 465 degrees Celsius (about 869 degrees Fahrenheit), which is hot enough to melt lead.
This makes Venus the hottest planet in the solar system, even hotter than Mercury, which is closer to the Sun.
The atmospheric pressure on Venus's surface is about 92 times that of Earth's at sea level, equivalent to the pressure found 900 meters underwater in Earth's oceans.
Despite these extreme conditions, scientists and space agencies have considered sending robotic missions to Venus.
The Soviet Union's Venera program successfully landed several spacecraft on Venus between 1970 and 1985, but none survived longer than 127 minutes.
Scientists are now exploring the possibility of a long-duration mission using high-temperature electronics.
Some researchers have even proposed building cloud cities in Venus's upper atmosphere, where temperatures and pressure are more Earth-like.
Venus rotates extremely slowly and in the opposite direction to most planets, meaning a day on Venus is longer than its year.
Its geological features include vast volcanic plains, highland regions, and thousands of volcanoes, some of which may still be active.
The planet lacks a magnetic field, making it vulnerable to the solar wind stripping away its atmosphere over billions of years.
Understanding Venus could give us crucial insights into the greenhouse effect and runaway climate change, processes that could affect Earth in the distant future.""",
                """GRADE 8 ESSAY RUBRIC — VENUS EXPLORATION
Score 6 (Excellent): Response demonstrates comprehensive understanding of Venus's extreme environment. Includes specific scientific details from the source text (temperature, pressure, atmospheric composition). Articulates at least three major challenges with technical reasoning. Organized with clear introduction, body paragraphs, and conclusion.
Score 5 (Proficient): Response demonstrates solid understanding of challenges with specific details from source text. Addresses most major challenges (temperature, pressure, atmosphere) with clear explanation. Organization is clear with minor weaknesses.
Score 4 (Adequate): Response addresses Venus exploration challenges with some specific details. May omit one major factor or lack precise data. Generally organized but may lack transitions.
Score 3 (Developing): Response addresses some challenges but is vague or relies on general statements. Limited use of source text details. Organization is present but inconsistent.
Score 2 (Beginning): Response identifies Venus as difficult to explore but provides few specific details. Minimal use of source text.
Score 1 (Minimal): Response demonstrates minimal understanding of Venus exploration challenges. Very few details from source text.
ASSIGNMENT: Write a detailed essay explaining the scientific and engineering challenges of exploring Venus based on the source text. Include specific data, address multiple challenge categories (thermal, atmospheric, pressure), and explain what innovations would be needed for a successful mission.""",
                [
                    "Venus presents an extraordinary scientific challenge that has captivated astronomers and engineers for decades. Based on the source text, the planet's extreme environmental conditions make it one of the most hostile destinations in our solar system, requiring unprecedented technological solutions before human or extended robotic exploration becomes feasible.\n\nThe most immediate and deadly challenge facing Venus exploration is the extreme surface temperature. According to the source text, temperatures on Venus average approximately 465 degrees Celsius, or 869 degrees Fahrenheit. This temperature exceeds the melting point of lead and is more than sufficient to destroy conventional spacecraft electronics within minutes of landing. The Soviet Venera landers, the only spacecraft to successfully touch down on Venus, survived for a maximum of just 127 minutes before being destroyed by the environment. Future missions would require revolutionary advances in high-temperature materials science, including specialized electronics capable of operating at temperatures far beyond what standard silicon components can withstand.\n\nThe atmospheric pressure presents an equally formidable engineering barrier. The source text notes that Venus's surface pressure is approximately 92 times that of Earth's sea level pressure, comparable to being submerged 900 meters underwater. Spacecraft structures must be designed to withstand this crushing force while simultaneously protecting sensitive instruments from corrosive sulfuric acid clouds. The combination of high pressure and high temperature places extraordinary demands on structural engineering, requiring materials that maintain integrity under conditions that would catastrophically fail conventional aerospace alloys.\n\nVenus's atmospheric composition introduces another layer of complexity. The dense carbon dioxide atmosphere, laced with sulfuric acid clouds, is chemically reactive and would rapidly corrode exposed metal surfaces. Scientists studying Venus's greenhouse effect have identified it as a cautionary example of runaway climate change, where the planet transitioned from potentially habitable conditions billions of years ago to its current hellish state. This scientific significance motivates continued research despite the enormous technological barriers.\n\nSome researchers have proposed innovative solutions to circumvent surface conditions entirely. The concept of aerial cloud cities, floating platforms in Venus's upper atmosphere where temperatures reach more Earth-like levels around 30-37 degrees Celsius, represents a creative engineering approach. At altitudes of roughly 50-60 kilometers, conditions are dramatically less hostile, potentially allowing for longer-duration scientific operations without the extreme thermal and pressure challenges of the surface.\n\nIn conclusion, exploring Venus demands solutions to interconnected challenges that no existing technology fully addresses. The combination of extreme heat capable of melting lead, crushing atmospheric pressure ninety-two times greater than Earth's, and a corrosive chemical environment creates an engineering challenge that will require decades of research and technological development. Understanding Venus, however, offers irreplaceable scientific insights into planetary evolution and climate dynamics that justify the considerable investment required.",
                    "The planet Venus, although often called Earth's twin due to its similar size, is actually one of the most hostile environments in the solar system. The source text reveals multiple scientific and engineering challenges that make exploring Venus extraordinarily difficult compared to missions to Mars or the Moon.\n\nFirst and foremost is the problem of temperature. Venus has an average surface temperature of 465 degrees Celsius, which is hot enough to melt lead. This extreme heat is caused by a runaway greenhouse effect driven by the thick carbon dioxide atmosphere. For any spacecraft or robotic explorer, this temperature presents an immediate and catastrophic risk. Normal electronic components fail at temperatures far below this level, requiring engineers to develop entirely new classes of heat-resistant materials and circuits.\n\nSecondly, the atmospheric pressure on Venus is crushing. At 92 times Earth's surface pressure, the environment would immediately destroy any spacecraft not specifically engineered to withstand it. This is equivalent to diving to 900 meters below the ocean surface. Soviet missions in the Venera program are the only spacecraft to have landed on Venus, and even those specially designed probes survived for at most two hours before being destroyed.\n\nThirdly, the composition of Venus's atmosphere is hazardous. The thick clouds of sulfuric acid would corrode many materials, adding chemical resistance requirements to the already demanding list of engineering specifications. Venus also lacks a magnetic field, exposing its surface to intense solar radiation.\n\nDespite these enormous challenges, scientists continue to study Venus because understanding its runaway greenhouse effect could yield crucial insights into climate change processes that may eventually affect Earth. Some proposals include floating stations in the upper atmosphere, where conditions are far less extreme, as a stepping stone toward eventual surface exploration.\n\nIn summary, exploring Venus requires overcoming extreme heat, crushing pressure, and corrosive atmospheric chemistry. These challenges demand revolutionary advances in materials science, thermal engineering, and spacecraft design before sustained exploration missions become possible.",
                ],
                [3.0, 5.0],
            ),
            (
                "Facial action coding system",
                """The Facial Action Coding System, commonly known as FACS, is a comprehensive, anatomically-based system for measuring all visually discernible facial movement.
Developed by Paul Ekman and Wallace Friesen in 1978, FACS breaks down facial expressions into individual components called Action Units (AUs), each corresponding to the contraction of specific facial muscles.
There are 46 primary Action Units in the core FACS coding system, along with additional codes for head position, eye movements, and other facial behaviors.
By combining different Action Units, virtually any facial expression can be described and documented with scientific precision.
FACS has become the standard tool in emotion research, enabling scientists to study the relationship between facial movements and emotional states across different cultures and populations.
The system has proven that six basic emotions — happiness, sadness, anger, fear, disgust, and surprise — are expressed through universal facial configurations recognized across cultures.
In clinical settings, FACS is used to detect deception, assess pain levels in patients who cannot verbally communicate, and diagnose neurological conditions affecting facial motor control.
Computer scientists and AI researchers have developed automated FACS coding systems using machine learning algorithms trained on large databases of annotated facial images.
These automated systems can process facial expressions in real time, enabling applications in human-computer interaction, mental health monitoring, and security screening.
FACS reliability depends heavily on trained human coders who must complete 100 or more hours of training to achieve certification, highlighting both the complexity and the precision of the system.""",
                """GRADE 8 ESSAY RUBRIC — FACIAL ACTION CODING SYSTEM
Score 6: Comprehensive analysis demonstrating deep understanding of FACS methodology and applications. Uses specific AU examples and details from source text. Analyzes both scientific and practical implications. Sophisticated organization with clear transitions.
Score 5: Solid analysis of FACS with multiple specific examples from source. Addresses methodology and applications with clear reasoning. Well-organized with minor gaps.
Score 4: Adequate explanation of FACS with some specific details. May focus on one aspect (e.g., only methodology OR only applications). Organized but lacks depth in analysis.
Score 3: Basic explanation of what FACS is with limited specific details. Organization present but may be list-like rather than analytical.
Score 2: Limited explanation with minimal source text details. May confuse or misrepresent specific aspects of FACS.
Score 1: Minimal response showing little understanding of FACS purpose or methodology.
ASSIGNMENT: Analyze how the Facial Action Coding System enables researchers to quantify human emotion. Explain the methodology, discuss its scientific contributions, and evaluate its applications in clinical and technological contexts based on the source text.""",
                [
                    "The Facial Action Coding System represents one of the most significant methodological advances in emotion research, providing scientists with a rigorous, anatomically-grounded framework for measuring and documenting facial expressions with unprecedented precision. By decomposing complex facial expressions into discrete, measurable components called Action Units, FACS transformed the study of human emotion from a largely subjective endeavor into a quantifiable scientific discipline.\n\nAt its core, FACS operates by mapping specific facial movements to the underlying muscular contractions that produce them. The system's 46 primary Action Units each correspond to the activation of particular facial muscles. For instance, AU6 represents the contraction of the orbicularis oculi muscle that raises the cheek, while AU12 activates the zygomaticus major muscle to pull the lip corners upward. By recording which Action Units are active in a given facial expression, researchers can document it with the same objectivity applied to physiological measurements.\n\nThis methodological precision has yielded transformative scientific insights. Paul Ekman's research using FACS provided empirical support for the theory of universal basic emotions, identifying six expressions — happiness, sadness, anger, fear, disgust, and surprise — that are recognized consistently across cultures that had never been in contact. This finding challenged purely social constructionist views of emotion and established a biological foundation for facial expression research.\n\nThe clinical applications of FACS are equally significant. In medical settings, trained FACS coders can assess pain intensity in patients who cannot verbally communicate, including infants, individuals with severe cognitive impairments, or those under general anesthesia. Neurologists use FACS to detect asymmetries in facial movement that may indicate stroke, Bell's palsy, or other conditions affecting cranial nerve function. In psychological research, FACS enables the identification of micro-expressions, fleeting facial movements lasting as little as 1/25th of a second that may reveal concealed emotional states.\n\nTechnological applications of FACS have expanded dramatically through machine learning. Automated FACS coding systems trained on large annotated image databases can now process facial expressions in real time, enabling emotion-aware computing applications. These systems are being deployed in driver monitoring to detect drowsiness, in customer service analytics to measure satisfaction, and in therapeutic contexts to help individuals with autism spectrum disorder recognize facial expressions.\n\nThe training requirements for human FACS coders — typically more than 100 hours to achieve certification — underscore the system's complexity and the precision required for reliable coding. This rigorous certification process ensures inter-rater reliability across research teams worldwide, making FACS-coded data comparable across studies and institutions.\n\nIn conclusion, FACS has fundamentally transformed emotion science by providing an objective, reproducible method for quantifying facial behavior. Its contributions span from foundational discoveries about universal human emotions to cutting-edge applications in artificial intelligence and clinical medicine, demonstrating that the precise measurement of facial movement unlocks profound insights into human emotional experience.",
                    "The Facial Action Coding System, or FACS, is a scientific tool developed by Paul Ekman and Wallace Friesen that allows researchers to systematically measure and categorize human facial expressions. By breaking expressions down into individual Action Units corresponding to specific facial muscle movements, FACS provides an objective method for studying human emotion that goes far beyond subjective description.\n\nThe primary scientific contribution of FACS is enabling the quantification of facial behavior. Before FACS, describing a facial expression relied on subjective interpretation that varied between observers. With FACS, a researcher can precisely document that a particular expression involves AU1 (inner brow raise), AU4 (brow lowerer), and AU17 (chin raiser), providing a complete and reproducible record. This precision transformed emotion research into a rigorous scientific discipline.\n\nFACS has been particularly valuable in cross-cultural research. Using FACS methodology, Ekman and colleagues demonstrated that six basic emotions produce universally recognized facial configurations across cultures worldwide. This finding contributed significant evidence to debates about the biological versus social origins of emotional expression.\n\nIn clinical applications, FACS provides tools for assessing pain and detecting neurological conditions. Pain researchers use FACS to measure facial expressions of discomfort in patients who cannot verbally report their experience. Neurologists analyze facial asymmetries coded in FACS to diagnose conditions affecting cranial nerve function. In security research, FACS-trained analysts examine micro-expressions to detect concealed emotional reactions.\n\nThe technological dimension of FACS has grown dramatically with artificial intelligence. Machine learning systems trained on FACS-coded facial image databases can now automatically recognize and code Action Units in real time, enabling applications from driver monitoring to mental health support tools.\n\nOverall, FACS represents a foundational methodological achievement that continues to enable scientific progress across emotion research, clinical medicine, and human-computer interaction.",
                ],
                [4.0, 3.0],
            ),
            (
                "The Face on Mars",
                """In July 1976, NASA's Viking 1 spacecraft was orbiting Mars, searching for a safe landing site for its companion lander, when it photographed a striking rock formation in the Cydonia region of Mars.
The image showed what appeared to be a giant human face staring up from the Martian surface, approximately 2 kilometers long and 260 meters high.
The image immediately captured public imagination, sparking widespread speculation that the feature might be an artificial structure built by an ancient Martian civilization.
NASA scientists at the time explained that the feature was almost certainly a natural mesa formation, and that the face-like appearance was the result of lighting conditions and the low resolution of the image.
Despite this official explanation, the Face on Mars became one of the most persistent and debated anomalies in space exploration history, fueling books, documentaries, and conspiracy theories for decades.
In 1998, the Mars Global Surveyor spacecraft captured new high-resolution images of the Cydonia region. These images clearly showed that the Face on Mars was indeed a natural geological formation — a mesa similar to those found in the American Southwest.
Further imaging in 2001 and 2006 by Mars Reconnaissance Orbiter provided even clearer views, confirming the mesa's natural origin and showing how the original Viking image's low resolution and oblique lighting had created the illusion of facial features.
Scientists use the Face on Mars as an educational example of pareidolia — the human brain's tendency to perceive familiar patterns, especially faces, in random or ambiguous stimuli.
The controversy surrounding the Face on Mars also highlights important lessons about scientific methodology, including the need for multiple independent lines of evidence, reproducibility, and skepticism toward extraordinary claims.""",
                """GRADE 8 ESSAY RUBRIC — THE FACE ON MARS
Score 6: Essay demonstrates sophisticated understanding of both the initial misunderstanding and scientific resolution. Uses specific details from source text. Analyzes pareidolia as a cognitive phenomenon. Discusses scientific methodology (burden of proof, multiple lines of evidence). Excellent organization.
Score 5: Essay explains initial misidentification and resolution with specific details. Addresses pareidolia or scientific skepticism. Well-organized with clear reasoning.
Score 4: Essay explains the story of the Face on Mars with some specific details. May not fully address the scientific reasoning behind initial misidentification or resolution. Organized but analysis is somewhat surface-level.
Score 3: Essay tells the basic story but relies on vague or general statements. Limited use of source text details or scientific reasoning.
Score 2: Essay mentions the Face on Mars with minimal specific detail or analysis. May contain inaccuracies or misrepresentations.
Score 1: Essay demonstrates minimal understanding of the topic with very limited detail.
ASSIGNMENT: Explain why the Face on Mars was initially misunderstood and what scientific evidence ultimately resolved the mystery. Discuss what this episode reveals about human perception and scientific methodology.""",
                [
                    "The story of the Face on Mars is one of the most compelling episodes in modern space exploration history, illustrating the powerful intersection of human psychology, scientific methodology, and the gradual accumulation of evidence that characterizes genuine scientific progress. The initial misidentification of a natural Martian mesa as an artificial structure, followed by its eventual scientific resolution, offers profound lessons about how humans perceive ambiguous stimuli and how science corrects its misconceptions over time.\n\nWhen Viking 1 photographed the Cydonia region in July 1976, the image conditions conspired to create a remarkably convincing illusion. The low resolution of the camera, combined with an oblique lighting angle that cast shadows emphasizing certain rock features, transformed a natural geological mesa into what appeared to be a human face roughly two kilometers in length. The brain's extraordinary sensitivity to facial patterns — a cognitive phenomenon scientists call pareidolia — caused both the public and even some researchers to perceive intentional structure in what was essentially a geological accident of shadow and shape.\n\nPareidolia is a deeply rooted cognitive mechanism with evolutionary origins. Humans evolved in environments where quickly recognizing faces was critically important for social survival, leading to neural systems that are biased toward detecting face-like patterns even in ambiguous or random stimuli. This is why people see faces in clouds, wood grain, and toast. The Viking image of the Cydonian mesa provided exactly the kind of ambiguous visual input that triggers pareidolia at a dramatic scale.\n\nThe scientific resolution of the mystery followed a methodologically exemplary path. Rather than dismissing the anomaly or declaring the case closed based on the initial official explanation, scientists continued to gather evidence as technology improved. When the Mars Global Surveyor captured high-resolution images of the Cydonia region in 1998, it provided dramatically clearer views that definitively showed the mesa's natural geological character. Subsequent imaging by the Mars Reconnaissance Orbiter in 2001 and 2006 provided additional confirmation from multiple angles and lighting conditions, satisfying the scientific requirement for independent replication.\n\nThis episode illustrates several important principles of scientific methodology. First, extraordinary claims require extraordinary evidence. The hypothesis that the Face on Mars was an artificial structure made by an alien civilization would have been one of the most significant discoveries in human history, requiring a correspondingly high evidentiary standard. Second, initial observations should be treated as hypotheses requiring testing, not conclusions. Third, technological advances enabling better measurements can and should be used to revisit unresolved questions.\n\nThe Face on Mars has become a standard educational example in discussions of critical thinking and scientific skepticism. It demonstrates that even intelligent, well-intentioned observers can be misled by ambiguous data, and that the corrective mechanism of science — independent verification, improved instrumentation, and open willingness to revise conclusions — is essential for distinguishing genuine discoveries from cognitive illusions.\n\nIn conclusion, the Face on Mars was initially misunderstood because the combination of low-resolution imaging, unusual lighting, and the human brain's powerful face-detection tendencies created a convincing illusion of artificial structure. Scientific resolution came through the systematic application of improved technology, independent verification, and the principle that claims must be proportionate to their evidence. The episode stands as a textbook example of how science corrects initial errors and how human perception can both inspire and mislead scientific inquiry.",
                    "The Face on Mars is an interesting case study in how human perception and scientific investigation can lead to very different conclusions about the same phenomenon. When Viking 1 took its famous photograph in 1976, it captured a rock formation that, due to the angle of sunlight and the camera's limited resolution, appeared to resemble a giant human face. This triggered widespread public excitement and speculation about ancient Martian civilizations.\n\nThe reason for the initial misunderstanding is primarily psychological. Scientists describe the human tendency to see faces in random patterns as pareidolia, a cognitive phenomenon where the brain's pattern recognition systems, which evolved to quickly identify faces for social reasons, perceive facial features in ambiguous visual data. The Viking image provided exactly the right combination of shadow, shape, and low resolution to strongly trigger this response.\n\nScientific methodology, however, demands more than a single ambiguous photograph. When NASA's Mars Global Surveyor sent back high-resolution images of the same region in 1998, the improved imagery clearly showed a natural mesa formation similar to geological features common in desert environments on Earth. Later missions confirmed these findings with even clearer images from multiple angles.\n\nThe resolution of the Face on Mars mystery demonstrates the importance of reproducible evidence and technological improvement in science. Rather than accepting the initial interpretation, scientists continued gathering evidence as better instruments became available. The multiple independent lines of evidence — from different spacecraft, different times, different lighting conditions — all pointed to the same natural geological explanation.\n\nThis episode teaches important lessons about both human psychology and scientific thinking. Our brains are wired to find patterns, especially faces, but science requires us to test these perceptions rigorously against objective evidence. The Face on Mars, once a source of speculation about alien civilizations, ultimately became a powerful educational tool for explaining pareidolia and the importance of critical thinking.",
                ],
                [5.0, 4.0],
            ),
            (
                "Driverless cars",
                """Autonomous vehicles, commonly known as driverless or self-driving cars, use a combination of sensors, artificial intelligence, and computational systems to navigate roads without human input.
The technology relies on LiDAR (Light Detection and Ranging), radar, cameras, GPS, and ultrasonic sensors to perceive the surrounding environment with far greater precision than human senses.
Machine learning algorithms, particularly deep neural networks, process this sensor data in real time to identify other vehicles, pedestrians, cyclists, lane markings, traffic signs, and potential hazards.
Proponents of autonomous vehicles argue that self-driving technology could dramatically reduce the approximately 1.35 million traffic fatalities that occur globally each year, since over 90 percent of accidents are caused by human error.
Major technology and automotive companies including Waymo, Tesla, Cruise, and Uber have invested hundreds of billions of dollars in developing and testing autonomous vehicle technology.
Despite significant progress, fully autonomous vehicles face major challenges, including navigating complex urban environments, handling adverse weather conditions like heavy rain or snow, and managing the ethical dilemmas inherent in unavoidable accident scenarios.
The so-called trolley problem — the moral dilemma of whether a self-driving car should sacrifice one person to save five — has become a central philosophical debate in autonomous vehicle development.
Cybersecurity represents a significant concern, as connected autonomous vehicles could potentially be hacked, exposing passengers and bystanders to life-threatening risks.
Regulatory frameworks for testing and deploying autonomous vehicles vary significantly across jurisdictions, creating legal uncertainty that has slowed widespread commercial deployment.
Economic disruption is another consideration: the widespread adoption of autonomous vehicles could eliminate tens of millions of driving-related jobs including truck drivers, taxi operators, and delivery personnel.""",
                """GRADE 10 ESSAY RUBRIC — DRIVERLESS CARS
Score 6: Essay provides balanced, sophisticated evaluation of both benefits and ethical concerns. Uses specific data from source text (accident statistics, company names, specific technologies). Analyzes the trolley problem or similar ethical dilemmas in depth. Addresses cybersecurity, economic disruption, or regulatory challenges. Excellent organization and sophisticated vocabulary.
Score 5: Essay evaluates benefits and ethical concerns with specific details from source. Addresses multiple aspects of the issue. Well-organized with clear reasoning and mostly specific evidence.
Score 4: Essay addresses both benefits and concerns but may be stronger in one area. Uses some specific details. Organized but analysis may be somewhat general.
Score 3: Essay addresses the topic with some relevant points but relies on general statements. Limited use of source text specifics. Organization present but analysis is shallow.
Score 2: Essay mentions autonomous vehicles with minimal specific detail. May focus only on benefits or only on concerns without balance.
Score 1: Essay demonstrates minimal understanding of the topic with very few relevant details.
ASSIGNMENT: Evaluate the benefits and ethical dilemmas of transitioning to fully autonomous driverless cars. Use specific evidence from the source text and address both the potential advantages and the significant risks and moral questions involved.""",
                [
                    "The transition to fully autonomous vehicles represents one of the most consequential technological shifts in modern transportation history, promising dramatic safety improvements while simultaneously raising profound ethical, economic, and security questions that society must carefully navigate. A balanced evaluation of this technology requires examining both its compelling benefits and its substantial challenges with equal rigor.\n\nThe safety case for autonomous vehicles is grounded in compelling statistical evidence. According to the source text, approximately 1.35 million people die in traffic accidents globally each year, with over 90 percent of these accidents attributable to human error. Autonomous vehicles, which cannot become distracted, fatigued, impaired by alcohol, or emotionally compromised, theoretically address the dominant cause of traffic mortality. If autonomous systems could even reduce accident rates by 50 percent, the global impact would be measured in hundreds of thousands of lives saved annually. This potential humanitarian benefit is the central moral argument in favor of pursuing autonomous vehicle development aggressively.\n\nHowever, the ethical landscape becomes dramatically more complex when we consider the moral dilemmas embedded in the software that governs these vehicles. The trolley problem — the philosophical question of whether an autonomous vehicle should take an action that harms one person to save five — is not merely an abstract thought experiment but a genuine software engineering decision. When unavoidable collision scenarios occur, the vehicle's algorithm must implement a predetermined response. Who decides whose life takes priority? Should the algorithm protect its passengers above all others, or should it minimize overall casualties regardless of who bears the cost? These questions expose the limitations of attempting to encode moral decisions in algorithmic form.\n\nCybersecurity threats represent a qualitatively different and potentially catastrophic risk category. Unlike human-driven vehicles, which can only be directly endangered by the errors of individual drivers, a connected autonomous vehicle fleet could theoretically be compromised simultaneously through software vulnerabilities. A successful cyberattack on autonomous vehicle software could cause multiple vehicles to behave erratically at once, potentially triggering mass casualty events. The source text identifies this as a significant concern, and security researchers have already demonstrated proof-of-concept attacks on connected vehicle systems.\n\nThe economic disruption potential of widespread autonomous vehicle adoption deserves serious consideration. The source text notes that tens of millions of driving-related jobs — including truck drivers, taxi operators, and delivery personnel — could be eliminated. While technological progress has historically created new categories of employment over time, the transition period could impose severe economic hardship on specific communities heavily dependent on transportation employment. Policymakers must develop proactive strategies to support workforce transitions rather than treating economic displacement as an inevitable externality of technological progress.\n\nThe regulatory framework challenges highlighted in the source text further complicate deployment timelines. Without consistent international standards for testing, certification, and liability, autonomous vehicle companies face a patchwork of legal environments that creates uncertainty and potentially unsafe incentives to deploy technology before it is fully validated.\n\nIn conclusion, the question of whether to transition to fully autonomous vehicles is not simply a matter of technological capability but a complex ethical, economic, and political decision. The potential to save over a million lives annually from traffic accidents represents a compelling moral imperative, but this benefit must be weighed against the genuine risks of algorithmic moral decision-making, cybersecurity vulnerabilities, economic disruption, and regulatory inadequacy. A responsible transition requires not just technological advancement but a societal conversation about the values we want embedded in our automated systems.",
                    "Driverless cars offer exciting potential to improve road safety and transportation efficiency, but they also present serious ethical and practical challenges that society must carefully address before widespread adoption.\n\nThe most compelling argument for autonomous vehicles is safety. The source text states that approximately 1.35 million people die in traffic accidents globally each year, with more than 90 percent caused by human error. Self-driving cars using LiDAR, radar, and advanced AI cannot become drunk, distracted, or tired, removing the dominant cause of accidents. This potential to save hundreds of thousands of lives annually is the strongest argument in favor of developing and deploying autonomous vehicle technology.\n\nHowever, the ethical challenges are significant. The trolley problem illustrates a genuine dilemma in autonomous vehicle programming. When an accident is unavoidable, the car's algorithm must decide how to respond. Should it protect its passenger at the cost of others, or minimize overall harm? There is no universally agreed ethical answer to this question, and encoding one particular moral framework into software without public debate raises serious concerns about who makes these life-and-death decisions.\n\nCybersecurity is another serious concern. Connected autonomous vehicles could be vulnerable to hacking, potentially allowing malicious actors to cause accidents or traffic disruptions at scale. The consequences of a successful cyberattack on an autonomous vehicle fleet could be catastrophic in ways that individual human driver errors cannot replicate.\n\nThe economic impact also deserves consideration. Millions of professional drivers could lose their jobs as autonomous vehicles become capable of performing transportation work. While new jobs may eventually emerge, the immediate disruption to communities dependent on driving employment would be severe without proactive policy responses.\n\nOverall, driverless cars have genuine potential to save lives and improve transportation, but realizing these benefits responsibly requires solving difficult ethical, security, and economic problems alongside the technical engineering challenges.",
                ],
                [4.0, 3.0],
            ),
            (
                "Does the electoral college work?",
                """The Electoral College is the formal body that elects the President and Vice President of the United States, established by Article II of the Constitution and modified by the Twelfth Amendment.
In the Electoral College system, each state is allocated a number of electors equal to its total congressional representation — the sum of its senators (always two per state) and its House representatives (proportional to population).
Currently, there are 538 total electoral votes, and a candidate must win at least 270 to become president.
Most states use a winner-take-all system, meaning the candidate who wins the popular vote in that state receives all of its electoral votes, regardless of the margin of victory.
Two states — Maine and Nebraska — use a congressional district method that can split their electoral votes between candidates.
Critics of the Electoral College argue that it distorts democracy by giving disproportionate political weight to voters in small states and by making presidential elections primarily about winning a small number of competitive swing states.
In the 2000 election, George W. Bush won the presidency while losing the national popular vote to Al Gore. In 2016, Donald Trump won the Electoral College while losing the popular vote to Hillary Clinton by approximately 2.9 million votes.
Defenders of the Electoral College argue that it preserves federalism by ensuring that candidates must build broad geographic coalitions rather than concentrating campaign resources in the most populous urban centers.
They also argue that the system encourages political stability by making it very difficult for third-party or regional candidates to win the presidency, preventing fragmented election outcomes.
Several reform proposals have been advanced, including the National Popular Vote Interstate Compact (NPVIC), which would commit participating states to award their electoral votes to the national popular vote winner if states representing at least 270 electoral votes join the compact.""",
                """GRADE 10 ESSAY RUBRIC — ELECTORAL COLLEGE
Score 6: Essay presents sophisticated, balanced analysis of arguments for and against the Electoral College with specific evidence from the source text (election examples, specific reform proposals, specific mechanisms). Demonstrates understanding of federalism and democratic theory. Presents and critiques multiple perspectives. Excellent organization and analytical writing.
Score 5: Essay analyzes arguments for and against with specific evidence. Addresses key aspects including recent election examples and reform proposals. Well-organized with clear reasoning.
Score 4: Essay discusses both sides with some specific evidence. May not address all major arguments or may be stronger in one direction. Organized but analysis has some gaps.
Score 3: Essay addresses the topic with some relevant points from source text but relies partly on general statements. Present both sides but without deep analysis.
Score 2: Essay discusses the Electoral College with minimal specific evidence. May focus primarily on one side of the debate.
Score 1: Essay demonstrates minimal understanding with very few relevant details.
ASSIGNMENT: Analyze the arguments for and against reforming or abolishing the United States Electoral College. Use specific evidence from the source text, address both perspectives fairly, and reach a reasoned conclusion about whether the system serves its democratic purpose effectively.""",
                [
                    "The Electoral College has been a source of democratic controversy and constitutional debate since its founding, but recent elections have intensified scrutiny of whether this distinctly American institution still serves the democratic values it was designed to protect. A rigorous analysis requires examining both the substantive arguments in its defense and the compelling case for reform with the same intellectual honesty, while recognizing that the question of what democratic legitimacy requires is ultimately a normative judgment that reasonable people can make differently.\n\nDefenders of the Electoral College ground their arguments primarily in federalism — the principle that the United States is a union of semi-sovereign states with distinct interests, not simply a single undifferentiated national electorate. By requiring presidential candidates to win electoral votes distributed across states rather than simply accumulating a national popular vote plurality, the system theoretically incentivizes candidates to build genuinely broad geographic coalitions rather than concentrating entirely on the most populous urban centers. This argument holds that democratic legitimacy in a federal republic requires not just numerical majority support but sufficient geographic breadth to demonstrate that a president represents the nation in its diversity.\n\nThe stability argument offers another substantive defense. By making it virtually impossible for third-party or regional candidates to win the presidency — since electoral votes must be won state-by-state — the Electoral College reinforces the two-party system that has historically produced relatively stable presidential transitions. Whether one views the two-party system as a feature or a bug of American democracy largely determines how compelling this argument appears.\n\nHowever, the critics of the Electoral College raise equally compelling concerns rooted in democratic equality. The winner-take-all allocation used by 48 states systematically amplifies the votes of residents in small states at the expense of residents in large states. A Wyoming resident's vote carries approximately three times the Electoral College weight of a California resident's vote, a mathematical distortion of the principle that democratic participation should be equal.\n\nThe most viscerally compelling argument against the Electoral College is its demonstrated capacity to produce presidents who lost the national popular vote. The elections of 2000 and 2016 both resulted in presidents who received fewer votes nationwide than their opponents — George W. Bush defeating Al Gore, and Donald Trump defeating Hillary Clinton by approximately 2.9 million votes. Critics argue that in a democracy, the candidate preferred by more citizens should win, and that a system that can routinely produce the opposite outcome has a fundamental legitimacy problem.\n\nThe National Popular Vote Interstate Compact offers an innovative reform path that does not require a constitutional amendment. By committing participating states to award their electoral votes to the national popular vote winner once states representing 270 electoral votes join, the NPVIC would effectively achieve a national popular vote outcome within the existing constitutional framework. As of recent analysis, states representing 209 electoral votes have joined, making this a realistic near-term possibility.\n\nIn conclusion, the Electoral College is a constitutionally embedded institution with substantive arguments in its defense rooted in federalism and stability, but it is also a system that demonstrably distorts democratic equality and has twice in recent memory produced presidents rejected by the national majority. The NPVIC represents a pragmatic reform path that deserves serious consideration as a means of aligning presidential elections more closely with democratic principles without requiring the extraordinarily high bar of a constitutional amendment.",
                    "The Electoral College is one of the most debated aspects of American democracy. The question of whether it still serves its intended purpose is complex, with serious arguments on both sides that deserve careful consideration.\n\nSupporters of the Electoral College argue that it protects the principle of federalism by ensuring that presidential candidates must win support across the diverse geography of the United States rather than simply appealing to the largest population centers. Under the current system, a candidate cannot win simply by running up large margins in California and New York while ignoring smaller states. This geographic breadth requirement, supporters argue, produces presidents with broader coalitions and more representative mandates.\n\nAdditionally, defenders argue that the Electoral College provides political stability by making it nearly impossible for third-party candidates to win the presidency. This two-party enforcement mechanism, though criticized for limiting voter choice, has historically produced clear electoral outcomes and smooth transitions of power.\n\nCritics, however, point to fundamental democratic equity problems. Because each state receives electoral votes equal to its congressional delegation, small states are mathematically overrepresented relative to their populations. A vote in Wyoming has approximately three times the electoral impact of a vote in California, which seems to contradict the democratic principle of equal representation.\n\nThe most serious criticism is the Electoral College's ability to produce presidents who lose the popular vote. This happened in 2000 when George W. Bush won despite Al Gore receiving more total votes, and again in 2016 when Donald Trump won while Hillary Clinton received approximately 2.9 million more votes nationwide. Critics argue this outcome directly contradicts democratic principles.\n\nReform proposals like the National Popular Vote Interstate Compact aim to address these concerns without requiring a constitutional amendment. Whether such reforms are adopted will determine whether the Electoral College evolves to meet modern democratic expectations or continues to operate as the Founders designed it, for better or worse.",
                ],
                [5.0, 3.0],
            ),
            (
                "Car-free cities",
                """Cities around the world have been experimenting with reducing or eliminating car traffic from urban centers, driven by concerns about air pollution, carbon emissions, public health, and urban livability.
The concept of car-free or car-reduced urban zones has been implemented to varying degrees in numerous European cities, including Amsterdam, Copenhagen, Oslo, and Ghent, as well as in cities across Latin America and Asia.
Amsterdam has built an extensive network of cycling infrastructure spanning over 800 kilometers of dedicated bike lanes, reducing automobile dependency and making cycling the primary mode of transportation for a significant portion of residents.
Oslo has virtually eliminated car traffic from its city center through a combination of removing parking spaces, expanding pedestrian zones, and investing heavily in public transit.
Studies have found that car-free policies produce measurable improvements in air quality, noise levels, and physical activity rates among urban residents.
However, converting urban areas to car-free environments also raises concerns about accessibility for people with disabilities, elderly residents, and those living in areas not well-served by public transit.
Small businesses in proposed car-free zones frequently oppose such policies initially, fearing that reduced car access will decrease customer foot traffic and harm their revenues.
Research suggests, however, that well-designed pedestrian zones often increase retail sales by making shopping environments more pleasant and encouraging longer visits.
Climate change has added urgency to the car-free city movement, as transportation accounts for approximately 24 percent of global carbon dioxide emissions, making urban vehicle reduction a significant component of emissions reduction strategies.
The COVID-19 pandemic created an unexpected natural experiment, with many cities expanding temporary pedestrian and cycling infrastructure during lockdowns, leading some to make these changes permanent after observing positive public health and economic outcomes.""",
                """GRADE 10 ESSAY RUBRIC — CAR-FREE CITIES
Score 6: Essay presents sophisticated analysis of social and environmental impacts using specific evidence from source text. Addresses multiple stakeholder perspectives including businesses, disability access, and environmental benefits. Analyzes trade-offs rather than presenting one-dimensional argument. References specific city examples and statistics. Excellent organization.
Score 5: Essay discusses multiple impacts with specific evidence from source. Addresses both benefits and challenges. References some specific examples. Well-organized with clear reasoning.
Score 4: Essay discusses impacts with some specific evidence but may focus more heavily on one category (social OR environmental). Organized but analysis has some gaps.
Score 3: Essay addresses the topic with some relevant points but relies partly on general statements without specific evidence from source text.
Score 2: Essay discusses car-free cities with minimal specific details. May present a one-sided argument without acknowledging trade-offs.
Score 1: Essay demonstrates minimal understanding with very few relevant details.
ASSIGNMENT: Discuss the social and environmental impacts of transforming modern urban centers into car-free cities. Use specific evidence from the source text, address multiple stakeholder perspectives, and evaluate the trade-offs involved in car-free urban transformation.""",
                [
                    "The movement toward car-free urban centers represents one of the most ambitious urban redesign initiatives of the twenty-first century, driven simultaneously by environmental imperatives and a reconception of what cities should fundamentally be — spaces designed for human flourishing rather than automotive convenience. The social and environmental impacts of car-free transformation are substantial, complex, and deeply intertwined, requiring careful consideration of both the compelling benefits and the genuine challenges of implementation.\n\nThe environmental case for car-free cities is grounded in the stark statistics of transportation's contribution to climate change. According to the source text, transportation accounts for approximately 24 percent of global carbon dioxide emissions, making it one of the largest contributors to greenhouse gas accumulation. Urban vehicles, concentrated in dense population centers, are responsible for a disproportionate share of this total, as well as generating nitrogen oxides, particulate matter, and other pollutants that directly harm respiratory health. Cities like Oslo and Amsterdam, which have implemented aggressive car reduction policies, have documented measurable improvements in air quality that translate directly into reduced rates of respiratory disease, particularly among children and elderly residents most vulnerable to air pollution.\n\nThe social benefits of car-free urban transformation extend significantly beyond air quality improvements. Studies consistently find that pedestrianized urban environments produce higher rates of physical activity, stronger social connections among residents, reduced noise pollution, and improved psychological well-being. Amsterdam's 800-kilometer network of dedicated bicycle infrastructure has transformed the city's relationship with transportation, making cycling the dominant mode of urban travel for a significant portion of residents and producing a public health dividend from increased physical activity. The COVID-19 pandemic provided an unexpected natural experiment in car-free infrastructure: cities that expanded temporary pedestrian and cycling zones during lockdowns observed such positive health and economic outcomes that many converted these temporary changes to permanent infrastructure.\n\nHowever, car-free transformation raises serious concerns about equity and accessibility that must be addressed for these policies to achieve genuine social justice alongside environmental goals. The source text appropriately highlights concerns about residents with disabilities and elderly individuals for whom car access may be essential to maintaining independent mobility. A car-free policy that improves quality of life for able-bodied cyclists while effectively excluding people with mobility impairments from city centers would represent a fundamental equity failure. Successful car-free cities must invest proportionally in accessible transit alternatives, including accessible bus and rail systems, adaptive cycling options, and maintained car access corridors for those with documented accessibility needs.\n\nSmall business concerns about car-free zones also deserve serious engagement rather than dismissal. The source text acknowledges that businesses in proposed car-free zones frequently oppose such changes initially, fearing reduced foot traffic. However, research cited in the source text suggests that well-designed pedestrian environments often increase retail sales by creating more pleasant shopping environments that encourage longer visits. This empirical finding does not eliminate the genuine risk that individual businesses may suffer during transition periods or that poorly designed pedestrian zones may fail to generate the expected foot traffic increases. Policymakers should consider transition support programs for affected businesses alongside infrastructure investment.\n\nThe examples of Oslo, Amsterdam, Copenhagen, and Ghent demonstrate that car-free or car-reduced urban environments are achievable in diverse city contexts, but each implementation reflects local geographic, economic, and political conditions that make direct replication challenging. Cities considering similar transformations must invest in comprehensive public transit networks before reducing car access, or risk creating transportation poverty for residents without alternatives.\n\nIn conclusion, the social and environmental impacts of car-free urban transformation are predominantly positive when implemented thoughtfully and equitably. The environmental benefits — reduced carbon emissions, improved air quality, and progress toward climate goals — are substantial and increasingly urgent. The social benefits — improved physical health, stronger community connections, and more livable urban environments — are equally compelling. The challenges of accessibility, equity, and economic transition are real but manageable with proactive policy design. The question is not whether car-free cities are desirable but whether societies have the political will to make the investments and accept the trade-offs required to achieve them.",
                    "Car-free cities offer significant social and environmental benefits but also raise important challenges related to accessibility, business impacts, and transportation equity that must be carefully addressed.\n\nFrom an environmental perspective, the case for reducing urban car traffic is compelling. The source text states that transportation accounts for approximately 24 percent of global carbon dioxide emissions. Urban vehicles also produce air pollutants that directly harm public health, particularly for vulnerable populations like children and the elderly. Cities like Oslo and Amsterdam have demonstrated that aggressive car reduction policies produce measurable improvements in air quality and corresponding health benefits.\n\nSocially, car-free urban zones can transform city life in positive ways. Research suggests that pedestrianized areas encourage physical activity, reduce noise pollution, and create more social, community-oriented public spaces. The COVID-19 pandemic showed that when cities expanded cycling and pedestrian infrastructure, many residents embraced these changes enthusiastically, leading some cities to make temporary pandemic-era changes permanent.\n\nHowever, significant challenges must be addressed. Accessibility for people with disabilities and elderly residents is a primary concern. Car-free policies must be accompanied by robust accessible transit alternatives to avoid excluding residents who depend on vehicle access for mobility.\n\nSmall businesses in proposed car-free zones often fear economic harm from reduced car access. While research suggests that well-designed pedestrian zones can actually increase retail sales by creating pleasant shopping environments, the transition period can be difficult for individual businesses, and policy support during conversion is essential.\n\nThe success of car-free cities ultimately depends on the quality and accessibility of alternative transportation systems. Without adequate public transit, cycling infrastructure, and accommodation for diverse mobility needs, car-free policies risk creating transportation inequality while failing to achieve their environmental goals. When implemented thoughtfully with strong transit investment, car-free transformation can deliver both environmental and social benefits.",
                ],
                [4.0, 3.0],
            ),
        ]

        data = []
        for idx, (topic, source, assign, essays, scores) in enumerate(topics, 1):
            for essay, score in zip(essays, scores):
                data.append({
                    "essay_id": 1000 + idx * 10 + essays.index(essay),
                    "prompt_name": topic,
                    "assignment": assign,
                    "source_text": source,
                    "full_text": essay,
                    "score": score,
                })

        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        print(f"✓ Generated ASAP 2.0 long-context fallback dataset: {len(df)} samples across {len(topics)} topics at {csv_path}")



    def load_data(self) -> pd.DataFrame:
        """Lazy load dataset into pandas DataFrame."""
        if self._df is None:
            self._df = pd.read_csv(self.csv_path)
        return self._df

    def get_samples(
        self,
        num_samples: int = 7,
        min_essay_len: int = 200,
        prompt_name: Optional[str] = None,
        seed: int = 42,
    ) -> List[Dict[str, Any]]:
        """Extract formatted essay evaluation samples across prompt categories.

        Args:
            num_samples: Number of samples to extract
            min_essay_len: Minimum word/character length filter for full_text
            prompt_name: Filter by specific prompt_name if specified
            seed: Random seed for sampling reproducibility

        Returns:
            List of sample dicts containing essay_id, prompt_name, prompt, student_essay, score, etc.
        """
        df = self.load_data()

        # Filter out rows missing essential text or score
        filtered_df = df.dropna(subset=["full_text", "score", "assignment"])
        filtered_df = filtered_df[filtered_df["full_text"].str.len() >= min_essay_len]

        if prompt_name:
            filtered_df = filtered_df[filtered_df["prompt_name"] == prompt_name]
            sample_df = filtered_df.sample(n=min(num_samples, len(filtered_df)), random_state=seed)
        else:
            # Sample evenly across distinct prompt_name categories
            distinct_prompts = filtered_df["prompt_name"].unique()
            per_prompt = max(1, num_samples // len(distinct_prompts))
            sampled_dfs = []
            for p in distinct_prompts:
                sub_df = filtered_df[filtered_df["prompt_name"] == p]
                sampled_dfs.append(sub_df.sample(n=min(per_prompt, len(sub_df)), random_state=seed))
            sample_df = pd.concat(sampled_dfs).head(num_samples)

        samples = []
        for _, row in sample_df.iterrows():
            # Combine available source texts
            source_texts = []
            for i in range(1, 5):
                col_name = f"source_text_{i}"
                if col_name in row and pd.notna(row[col_name]):
                    source_texts.append(str(row[col_name]).strip())
            
            combined_source = "\n\n".join(source_texts)
            
            # Format long context prompt
            prompt_text = self.format_long_context_prompt(
                source_text=combined_source,
                assignment=str(row["assignment"]).strip(),
                student_essay=str(row["full_text"]).strip(),
            )

            samples.append({
                "essay_id": str(row["essay_id"]),
                "prompt_name": str(row.get("prompt_name", "Unknown")),
                "score": float(row["score"]),
                "formatted_prompt": prompt_text,
                "student_essay": str(row["full_text"]).strip(),
                "assignment": str(row["assignment"]).strip(),
            })

        return samples


    @staticmethod
    def format_long_context_prompt(
        source_text: str,
        assignment: str,
        student_essay: str,
    ) -> str:
        """Format a structured long-context prompt for LLM evaluation."""
        prompt = (
            "You are an expert Automated Essay Scoring (AES) evaluator.\n\n"
            "### SOURCE READING TEXT\n"
            f"{source_text if source_text else 'No additional reading text provided.'}\n\n"
            "### GRADING ASSIGNMENT & RUBRIC\n"
            f"{assignment}\n\n"
            "### STUDENT ESSAY TO EVALUATE\n"
            f"{student_essay}\n\n"
            "### EVALUATION INSTRUCTION\n"
            "Based on the rubric above, evaluate the student essay and respond with ONLY the score.\n"
            "Your response MUST start with exactly this format: 'Score: X'\n"
            "where X is a single integer between 1 and 6.\n"
            "Example: Score: 4\n"
            "Do NOT write anything before 'Score:'. Begin your response with 'Score:' immediately."
        )
        return prompt

