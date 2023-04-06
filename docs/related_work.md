# Related work

## Index

- [Article 1](#article-1) CNN for task classification using computer screenshots for integration into dynamic calendar/task management systems
- [Article 2](#article-2) Guess What's on my Screen? Clustering Smartphone Screenshots with Active Learning
- [Article 3](#article-3) Understanding Screen Relationships from Screenshots of Smartphone Applications
- :star: [Article 4](#article-4) Screenomics: A New Approach for Observing and Studying Individuals’ Digital Lives
- :star: [Article 5](#article-5) Screenomics: A Framework to Capture and AnalyzePersonal Life Experiences and the Ways thatTechnology Shapes Them 

## Article 1

[Link to the article](http://cs231n.stanford.edu/reports/2015/pdfs/anand_avery_final.pdf)

**Title:** CNN for task classification using computer screenshots for integration into dynamic calendar/task management systems.<br>

**Authors:** Anand Sampat, Avery Haskell - Stanford University<br>

**Context:** This is an unpublished manuscript of 2015 that appears to be cited in 5 other published articles according to Google Scholar.<br>

**Summary:** This project aims to classify high level tasks (e.g. ‘checking emails’, ‘photo editing’, ‘social media’) a user does on his or her computer using deep CNN networks. The technique of transfer is implemented using scene-classification pretrained models which provide weights that significantly reduce training time by placing the solver in a state closer to the optimal state. It explores two datasets – one small (1900 training examples), and one larger (10,800 training images). Moreover, additional work in window detection and streaming data (time component) is carried out.<br>

**Core idea:**
1. Elaboration of a new screenshot labeled dataset using automated image search from search engine scraping and screenshots taken at regular intervals from a computer.
2. Overcoming data challenges: Data sparsity with data augmentation (moving windows); Data Diversity by avoiding too homogeneous data (e.g. OS, theme, color scheme); Data Cleaning by hand curation, duplicate removal and filters.
3. The backbone architecture of the model is the Hybrid CNN. This architecture is simplified by reducing the number of layers. The architecture was pre-trained on the Places205CNN dataset from MIT and by freezing some of the layers and training only the last 3 to 5 layers the new model is obtained.
4. Finally, work on data interpretation is also presented. As a result of these observations the papers explores two possibilities:
    - a) Window detection: using bounding boxes to capture different windows in the screen (OpenCV library: canny edge detection).
    - b) Streaming data: aiming to capture the idea that temporal context is important to classification by using more than one image (10 sec intervals) to make a prediction.
<br>

**Conclusion:**
- On a large dataset the best model performed reasonably with 64% accuracy using a CNN with weights from hybridCNN training and retraining the last conv layer and the two fully-connected layers on 10,800 images. 
- Despite not showing accurate results, the paper suggests that in order to better integrate this, data streaming is very important. Furthermore, obtaining contextual cues via segmentation of the image with bounding boxes can be used in conjunction with a contextually constrained deep network to improve task labeling.



## Article 2 
**(in progress...)**

[Link to the article](https://arxiv.org/pdf/1805.07964.pdf)

**Title:** Guess What's on my Screen? Clustering Smartphone Screenshots with Active Learning

**Authors:** Agnese Chiatti, Dolzodmaa Davaasuren, Nilam Ram, Prasenjit Mitra, Byron Reeves, Thomas Robinson - The Pennsylvania State University & Stanford University

**Context:** This paper was published in January 2019 as a collaboration between The Pennsylvania State University and Stanford University. Stanford University involved their [Screenomics Lab](https://screenomics.stanford.edu/) members for useful discussions and acknowledge the data and computational support provided for the experiments.
<br>

**Summary:**

<br>

**Core idea:**

<br>

**Conclusion:**


## Article 3
**(in progress...)**

[Link to the article](https://dl.acm.org/doi/pdf/10.1145/3490099.3511109)

**Title:** Understanding Screen Relationships from Screenshots of Smartphone Applications

**Authors:**  Shirin Feiz, Jason Wu, Xiaoyi Zhang, Amanda Swearngin, Titus Barik, Jeffrey Nichols

**Context:**

<br>

**Summary:**

<br>

**Core idea:**

<br>

**Conclusion:**

## Article 4

[Link to the article](https://journals.sagepub.com/doi/pdf/10.1177/0743558419883362)
**Title:** Screenomics: A New Approach for Observing and Studying Individuals’ Digital Lives\
**Authors:** Nilam Ram , Xiao Yang, Mu-Jung Cho, Miriam Brinberg, Fiona Muirhead, Byron Reeves, and Thomas N. Robinson\
**Keywords:** screenomics, screenome, smartphone, social media, adolescence, digital media, intensive longitudinal data, experience sampling\
**Abstract:** The article presents "screenomics," a new approach for studying individuals' digital experiences. The study analyzed over 500,000 smartphone screenshots from four Latino/Hispanic youth aged 14 to 15 years from low-income, racial/ethnic minority neighborhoods using computational machinery and machine learning algorithms. The results show how adolescents' digital lives differ across persons, days, and hours, highlighting their switching among multiple applications and exposure to different content. The authors suggest that screenomics can provide more detailed data for studying digital lives and testing theories about media use and development. \
**Takeaways:**
- "Consolidation of technology embedded into smartphone devices mean
that it is now possible to switch between radically different content on a single screen on the order of seconds—for example, when an individual switches
from watching a cat video on YouTube to taking, editing, and sharing a selfie
in Snapchat to searching for a restaurant with Google to texting a friend to
arrange a time to meet. Self-reports of media exposure and behavior simply
do not provide accurate representation of digital lives that weave quickly—
often within a few seconds—between different software, applications, locations, functionality, media, and content. In contrast, the screenome—made of
screenshots taken frequently at short intervals—is a reconstruction of digital
life that contains all the information (i.e., the specific words and pictures) and
details about what people are actually observing, how they use software (e.g.,
writing, reading, watching, listening, gaming), and whether they are producing information or receiving it. The screenshots capture activity about all
applications and all interfaces, regardless of whether researchers can establish connections to the sources of information via software logging or via
relationships with businesses that own the logs (e.g., phone records, Facebook,
and Google APIs). Sequences of screenshots (i.e., screenomes) provide the
raw material and time-series records needed for fine-grained quantitative and
qualitative analysis of fast changing and highly idiosyncratic digital lives"
- "In brief, each screenshot is “sequenced” using a custom-designed module wrapped around open-source tools for image and document processing. Screenshots are:
    1. Converted from RGB to grayscale
    2. Binarized to discriminate foreground from surrounding background
    3. Segmented into blocks of text (OCR) and images, and passed through a recognition engine (Tesseract/OpenCV-based) that identifies text, faces, logos, and objects.
        - **word count**: a simple
description of quantity of text, takes a bag-of-words approach and is calculated
for each screenshot as the number of unique “words” (defined as any string of
characters separated by a space) that appeared on the screen
        - **Word velocity**, a
description of how quickly the content is moving (e.g., when scrolling through
text), is calculated as the difference between the bag-of-words obtained from
two consecutive screenshots, specifically the number of unique new words that
did not appear in the prior screenshot.
        - **sentiment analysis**
        - other language characteristics (e.g., complexity)
        - a **bagof-words analysis of the text** (LIWC, Pennebaker, Chung, Ireland, Gonzales, &
Booth, 2007) provided 93 variables that indicated (through dictionary lookup)
prevalence of first-person pronouns, social, health, and other types of words.
        - **Image complexity** was computed as the across-pixel heterogeneity (i.e.,
“texture”), as quantified by entropy of an image across 256 gray-scale colors
        - **Image velocity**, a description of visual flow, was then calculated for each screenshot as the change in image complexity from the prior
screenshot. The velocity measure (formally, the first derivative of the image
complexity time-series) indicates how quickly the content is changing (e.g.,
by analogy, how quickly a video shot pans across a scene) and is known to
influence viewers’ motivations and choices
        - **Logos are identified using template matching methods**
        - Together, the textual and graphical features provide a quantitative description of each screenshot that was used in subsequent analysis.
    4. The resulting ensemble of text snippets and image data (e.g., number of faces) are then compiled into Unicode text files, one for each screenshot, that are integrated with metadata within a document-based database that facilitates storage, retrieval (through a local API), and visualization"
- **Fine-Grained Temporal Granularity:** The high frequency of screenshot sampling allows for analysis at multiple
time scales, from seconds to hours to days to months (Ram & Reeves, 2018).
One may zoom in or out across time scales to examine specific behaviors of
finite length, cyclic or acyclic patterns of behavior, or long-term trends in
behavior that match the times scale relevant to theoretical questions (Aigner,
Miksch, Muller, Schumann, & Tominski, 2008). The screenome includes
data about particular moments in life, repeated exposures and behaviors, and
the full sequence of contiguous, intermittent, and potentially interdependent
digital experiences.
- "The screenome is the
only data that we know of that preserves the sequencing and content of all
experiences represented on digital devices and opens them up for study and
analysis of within-person behavior change."
- :star: "In parallel, selected subsets of the screenshots were used to identify and
develop schema that provided meaningful description of the media that
appeared on the screen. For example, screenshots can be tagged with codes that
map to specific behavioral taxonomies (e.g., emailing, browsing, shopping)
and content categories (e.g., food, health). Manual labeling of large data sets
often use public crowd-sourcing platforms (e.g., Amazon Mechanical Turk;
Buhrmester, Kwang, & Gosling, 2011), but the confidentiality and privacy protocols for the screenome require that labeling be done only by members of the
research team who are authorized to view the raw data. Through a secure
server, screenshot coders were presented with a series of screenshots (random
selection of sequential chunks that provided coverage across participants and
days), and they labeled the action and/or content depicted in each screenshot
using multicategory response scales (e.g., Datavyu Team, 2014) or open-ended
entry fields (QSR International Pty Ltd, 2012). These annotations were then
used both to describe individuals’ media use, and as ground truth data to train
and evaluate the performance of a collection of machine learning algorithms
(e.g., random forests) that used the textual and graphical features (e.g., image
complexity) to replicate the labeling and extend it to the remaining data."
- :star: Applications. To describe the applications that adolescents used, we manually
labeled a subset of 10,664 screenshots with the name of the specific applications appearing on the screen. Five labelers were tasked with generating
ground truth data for the machine learning algorithms. This team’s labeling
was evaluated using standard calculation of interrater reliability (κ > .90) on
test data. Discrepancies were resolved through group discussion and look-up of potential categorization by focused search of web-based information until
the matching application was identified. This ground-truth data were then
used to train a machine learning algorithm to accurately identify the application based on up to 121 textual and graphical features already extracted from
each screenshot. After multiple iterations, we obtained a random forest with
600 trees (Hastie, Tibshirani, & Friedman, 2001; Liaw & Wiener, 2002) that
had an out-of-bag error rate of only 6.8%, and used that model to propagate
the app labels to the rest of the data. Following previous research (Böhmer,
Hecht, Schöning, Krüger, & Bauer, 2011; Murnane et al., 2016), apps were
further categorized by type using the developer-specified category associated
with the app in the official Android, Google Play marketplace. Thus, each
screenshot was labeled with both the specific application name (e.g., Snapchat, Clock, Instagram, Youtube) and the more general application type. Represented types included Comics, Communication, Education, Games, Music
& Audio, Photography, Social, Study (our data collection software), Tools,
and Video Players & Editors.
- Production/consumption. Screenshots capture moments when individuals are
producing content; for example, when using the keyboard to type a text message. To identify such behavior, we manually labeled 27,000 screenshots (in
our larger repository) with respect to whether the user was producing or consuming content. These ground-truth data were then used to train an extreme
gradient boosting model (Chen, He, & Benesty, 2015; Friedman, 2001) that
accurately classified screenshots as production or consumption. After tuning,
through grid search of hyperparameters and tenfold cross-validation, performance reached 99.2% accuracy. Using this model, all screenshots were
labeled as an instance of production or consumption
- **Sentiment**. The text that appears in any given screenshot can be characterized
with respect to emotional content or sentiment (Liu, 2015). Here, we used the
modified dictionary look-up approach implemented in the sentimentr package in R (Rinker, 2018). This implementation draws from Jockers’s (2015)
and Hu and Liu’s (2004) polarity lexicons to first identify and score polarized
words (e.g., happy, sad), and then makes modifications based upon valence
shifters (e.g., negations such as not happy). A sentiment score was calculated
for each screenshot based on all the alphanumeric words extracted via optical
character regocnition (OCR) from each screenshot. Sentiment scores greater
than zero indicate that the text in the screenshot had an overall positive
valence, and scores less than zero indicate an overall negative valence.
Altogether, our combined use of feature extraction tools (e.g., OCR of
text, quantification of image complexity, logo detection), machine learning
algorithms, human labeling, and qualitative inquiry provided for rich idiographic description of the digital experiences of these adolescents’ everyday
smartphone use.
\[**Limitations and Future Directions**]
- Second, although we have all the content that appeared on the smartphone
screen, we only extracted a relatively small set of features from each screenshot
(e.g., complexity of image), only quantified sentiment of text using a single
dictionary (e.g., not sentiment of images), and only examined one type of content (food-related images and text). The analysis pipeline and procedures
developed here are being expanded (see also Reeves et al., 2019). On-going
research is expanding the feature set (e.g., using NLP, object recognition, image
clustering); expanding the perspectives from which the data are approached
(e.g., cognitive costs of task switching, language of interpersonal communication) and the types of content examined (e.g., politics, finances, health); and
examining how general and specific aspects of the screenome are related to and
interact with individuals’ other time-invariant characteristics and time-varying
behaviors.
- Third, we obtained screenshots every 5 seconds that the participants’
screens were in use during a 30- to 100-day span. While the sampling of screen
content every 5 seconds provided unprecedented detail about actual smartphone use that is not available in any other studies that we know of, the fine
granularity of the data still misses some behaviors. Our “slow frame rate”
sampling misses “quick-switches” that occur when, for instance, an individual
moves through multiple social media platforms in rapid succession as they
check what’s new, or are engaged in synchronous text message conversations
with multiple individuals simultaneously. Although costly in terms of data
transfer, storage, and computation, study of some phenomena might require
faster sampling that is closer to continuous video.
- Finally, we note that although our analysis made use of a variety of contemporary quantitative and qualitative methods, it is descriptive. We have no yet used the intensive time-series data to build and test process models of
either moment-to-moment behavior (e.g., task switching). The initial explorations here
have, however, developed a foundation for future data collection and for
analyses that make use of mathematical and logical models that support inferential and qualitative inference to many individual-level and population-level
phenomena. We look forward to the challenges and new insights yet to come.
One of the main challenges we face is the need and cost (mostly in terms of
time) of human labeling. The machine learning algorithms we used here for
identifying the application being used in each screenshot was based on a
subset of 10,664 screenshots that were labeled by our research team. Future
work will eventually require more labeling. While many big-data applications make use of crowd-sourcing platforms such as Mechanical Turk to label
data relatively cheaply, these platforms are not viable for labeling of screenome data. Given the potentially sensitive and personal nature of the data,
very strict protocols are in place to ensure that the data remain private. Thus,
labeling and coding can only be done in secure settings by approved research
staff. Our protocols underscore that the richness of the data come with social
responsibility for protection of the data contributors.
<br>

## Article 5

[Link to the article](https://www.tandfonline.com/doi/full/10.1080/07370024.2019.1578652)
**Title:** Screenomics: A Framework to Capture and Analyze Personal Life Experiences and the Ways that Technology Shapes Them\
**Authors:** Byron Reeves,Nilam Ram,Thomas N. Robinson,James J. Cummings,C. Lee Giles,Jennifer Pan,Agnese Chiatti,Mj Cho,Katie Roehrick,Xiao YangORCID Icon,Anupriya Gagneja,Miriam Brinberg,Daniel Muise,Yingdan Lu,Mufan Luo,Andrew Fitzgerald &Leo Yeykelis\
**Takeaways:**
- **Screenome:** The authors propose a new approach called "screenomics" for studying individuals' digital experiences, specifically their unique "screenome," which is the record of experiences on digital devices with screens. The screenome is composed of smartphone, laptop, and cable screens, with information sequences describing the temporal organization, content, functions, and context of person-screen interactions. It can be linked to other levels of analysis, showing how biological and cultural factors may interact with digital experiences. The screenome can provide insights into unique social, psychological, and behavioral characteristics and experiences related to individuals.
- Screenome workflow: ![Screenome](ScreenomeWorkflow.png)
- **Feature extraction**:
    - **OCR**: A major component of screenshot content is text. Some of the challengestypically associated with text extraction from degraded or natural images (e.g.,diverse text orientation, heterogeneous background luminance) are not problematicwith screenshots. But some are including inconsistency in fonts, screen layouts, andpresence of multiple overlapping windows and these problems complicate identifi-cation, extraction, and organization of textual content. Our current text extractionmodule (Chiatti et al.,2017) makes use of open-source tools: OpenCV for imagepre-processing (Culjak, Abram, Pribanic, Dzapo, & Cifrek,2012), and Tesseract forOCR (Smith,2007). As shown inFigure 1, each screenshot is first converted from RGB tograyscale and then binarized to discriminate the textual foreground fromsurrounding background. Simple inverse thresholding combined with Otsu’sglobal binarization technique (Otsu,1979) has been sufficient, given thatmost screenshots have consistent illumination across the image. Candidateblocks of text are then identified using a connected component approach(Talukder & Mallick,2014) where white pixels are dilated, and a rectangularcontour (i.e., bounding box) is wrapped around each region of text. Given thepredominantly horizontal orientation ofscreenshot text, processing efficiency ismaintained by skipping the skew estimation step. Each candidate block of textis then fed to a Tesseract-based OCRmodule to obtain a collection of textsnippets that are compiled into Unicode text files, one for each screenshot. Ourpublished studies, wherein we compared OCR results against ground-truthtranscriptions of 2,000 images, show the accuracy of the text extraction proce-dures at 74% at the individual character level (Chiatti et al.,2017). OngoingScreenomics167experiments support further improvements through the integration of neuralnet-based line recognition that is trained and tuned specifically on the expand-ing screenshot repository, similar to the approach used in the OCRopus frame-work (Breuel,2008), and included in the alpha version of Tesseract 4.0.Improvements in image segmentation, in particular, are expanding furtheropportunities for natural language processing analyses (e.g., LIWC; Pennebaker,Booth, Boyd, & Francis,2015) that are then used to identify meaningfulcontent from the extracted text.
    - Labeling (Human Tagging): There are some features of screenshots that are of theoretical interest but for which there are not yet automated methods for obtaining labels. Consequently, we facilitate labeling of individual screenshots with tools for human tagging. Humanlabeling of big data often uses public crowd-sourcing platforms (e.g., AmazonMechanical Turk; Berinsky, Quek, & Sances,2012; Bohannon,2011; Buhrmester,Kwang, & Gosling,2011; Horton, Rand, & Zeckhauser,2011). Confidentiality andprivacy protocols for the screenome require that labeling is done only by membersof the research team that are authorized to see the raw data. Manual labeling and text transcription are done using a custom module built on top of the localturkopensource API (Vanderkam,2017) and for some tasks the opensource Datavyu(2014) software.\
    \
    [**LIMITATIONS**]
    - Missing screens
    - Non-Screen Behavior
    - System Optimization \
    [**EXTENSIONS**]
    - Structured Vs. Unstructured Data
    - Creating Behavioral Indices from Screenome Data: For example, what might be the pace of screen switching for different personalities?
    - Internal and External Validity: ll of our data so far have been generated by adult volunteers recruited for theirwillingness to participate in the new research. We have deliberately recruited digitallyactive younger adults, and although the samples are balanced by gender and geography,they are not randomly selected or representative of national populations or subgroups interms of race, income, and other important characteristics that may influence technologyuse. Consequently, we have prioritized internal validity and demonstrations of the value ofscreenome data. Going forward, the samples will need to be more representative. Thereare significant known differences in how individuals use digital media (Correa et al.,2010;Jackson et al.,2008), and those differences must be reflected in future samples.
    - Between-Person Differences and Within-Person Change
    - Time-Scales: All of the screenomes reported here were constructed using screenshots takenat 5-s intervals. That interval, which is considerably faster than many experiencesampling techniques used in research, was only a first attempt to define the bestsampling frequency. The switching time results (Section 5.1) suggest using evenshorter intervals. Indeed, with 5-s intervals, we are only able to model behaviorsmanifesting at a ten-second time-scale. Screenshots taken every one or 2 s couldbetter characterize, for example, quick switches between sending and receipt ofScreenomics189messages in a synchronous text exchange or observations of how swiping orpinching content accelerates and decelerates across context and time.
- :star: **Use of the Screenome for Interventions**: One promising future use of the system and approach presented here is theability to“interact”with an individual’s screenome and to deliver interventions thatalter how people think, learn, feel and behave. This may help realize the promise ofprecision interventions to preempt or treat unwanted thoughts, emotions or beha-viors, and to promote desirable ones. Delivering the right intervention to the rightperson at the right time and in the right context with the lowest risks of adverse sideeffects could close the loop between feedback and optimization in real time. Someof the most exciting potentials for precision interventions are in health. Many healthparameters are dynamic, in that they change and vary over time (e.g., bloodpressure). The screenome may allow researchers to identify causal relations ata timescale that matches the speed at which symptoms and diseases actually vary.
- **Privacy**: ... The screenome project collected data using strict privacy protocols and the data are securely stored and viewed only by trained staff in the lab. However, some people declined to participate due to concerns about privacy, particularly regarding text messages. Two-thirds of people accepted participation, higher than previous research, but additional research is needed to address privacy issues. The screenome framework is suggested as a new way to study human behavior and technology's impact, and data transfer risks can be reduced through local analysis and transferring only summary results.
    
## Other references


1. T. Kekec, R. Emonet,E. Fromont,A. Tremeau,and C. Wolf ”**Contextually Constrained Deep Networks for Scene Labeling**”. [http://www.bmva.org/ bmvc/2014/files/paper032.pdf]
2. B. Zhou, A. Lapedriza, J. Xiao, A. Torralba, and A. Oliva. ”Learning Deep **Features for Scene Recognition** using Places Database.” Advances in Neural Information Processing Systems 27 (NIPS), 2014 [http: //places.csail.mit.edu/]
3. A. Krizhevsky,I. Sutskevar, G. Hinton. **ImageNet Classification with Deep Convolutional Neural Networks**. Advances in Neural Information Processing Systems 25 (NIPS) 2012 [http://papers.nips.cc/paper/ 4824-imagenet-classification-with-deep-convolutpdf]
4. J. Long,E. Shelhamer, and T. Darell. **Fully Convolutional Networks for Semantic Segmentation**.[http: //arxiv.org/pdf/1411.4038v1.pdf]
5. **Canny edge detection** (OpenCV) http://opencv-python-tutroals. readthedocs.org/en/latest/py_ tutorials/py_imgproc/py_canny/py_ canny.html
6. M. D. Zeiler and R. Fergus. ”Visualizing and Understanding Convolutional Networks. [http:// arxiv.org/pdf/1311.2901v3.pdf]
7. Chiatti, A., Yang, X., Brinberg, M., Cho, M. J., Gagneja, A., Ram, N., … Giles, C. L. (2017). Text extraction from smartphone screenshots to archive in situ media behavior. Proceedings of the Ninth International Conference on Knowledge Capture, Austin, TX. ACM.. [Crossref], [Google Scholar]
8. Chittaranjan, G., Blom, J., & Gatica-Perez, D. (2013). Mining large-scale smartphone data for personality studies. Personal and Ubiquitous Computing, 17(3), 433–450. doi:10.1007/s00779-011-0490-1 [Crossref], [Web of Science ®], [Google Scholar]
9. Brown, J. S. (2000). Growing up: Digital: How the web changes work, education, and the ways people learn. Change: the Magazine of Higher Learning, 32(2), 11–20. doi:10.1080/00091380009601719 [Taylor & Francis Online], [Google Scholar]
10. Breuel, T. M. (2008). The OCRopus open source OCR system. In Proc. SPIE 6815, document recognition and retrieval XV (Vol. 6815, pp. 68150F1–68150F15). Bellingham, WA: International Society for Photonics and Electronics. [Crossref], [Google Scholar]
11. Vogelsang, Andreas, and Markus Borg. "Requirements engineering for machine learning: Perspectives from data scientists." 2019 IEEE 27th International Requirements Engineering Conference Workshops (REW). IEEE, 2019.
12. Reeves, Byron, Thomas Robinson, and Nilam Ram. "Time for the human screenome project." Nature 577.7790 (2020): 314-317.
13. Suatap, Chayanin, and Karn Patanukhom. "Game genre classification from icon and screenshot images using convolutional neural networks." Proceedings of the 2019 2nd Artificial Intelligence and Cloud Computing Conference. 2019.
14. Ram, Nilam, et al. "Screenomics: A new approach for observing and studying individuals’ digital lives." Journal of Adolescent Research 35.1 (2020): 16-50.
15. Reeves, Byron, et al. "Screenomics: A framework to capture and analyze personal life experiences and the ways that technology shapes them." Human–Computer Interaction 36.2 (2021): 150-201.
16. Abdali, Sara, et al. "Identifying Misinformation from Website Screenshots." Proceedings of the International AAAI Conference on Web and Social Media. Vol. 15. 2021.
17. Cockburn, A., & McKenzie, B. (2001). What do web users do? An empirical analysis of web use. International Journal of Human-Computer Studies, 54(6), 903–922. doi:10.1006/ijhc.2001.0459 [Crossref], [Web of Science ®], [Google Scholar]
18. Culjak, I., Abram, D., Pribanic, T., Dzapo, H., & Cifrek, M. (2012, May). A brief introduction to OpenCV. MIPRO, 2012 Proceedings of the 35th international convention, Opatija, Croatia, 2012 (pp. 1725–1730). IEEE. [Google Scholar]
19. Dingler, T., Agroudy, P. E., Matheis, G., & Schmidt, A. (2016, February). Reading-based screenshot summaries for supporting awareness of desktop activities. In Proceedings of the 7th Augmented Human International Conference 2016 (pp. 27). New York, NY: ACM. [Crossref], [Google Scholar]
20. Do, T. M. T., & Gatica-Perez, D. (2014). The places of our lives: Visiting patterns and automatic labeling from longitudinal smartphone data. IEEE Transactions on Mobile Computing, 13(3), 638–648. doi:10.1109/TMC.2013.19 [Crossref], [Web of Science ®], [Google Scholar]
21. Hembrooke, H., & Gay, G. (2003). The laptop and the lecture: The effects of multitasking in learning environments. Journal of Computing in Higher Education, 15(1), 46–64. doi:10.1007/BF02940852 [Crossref], [Google Scholar]
22. Kumar, R., & Tomkins, A. (2010, April). A characterization of online browsing behavior. In Proceedings of the 19th international conference on World Wide Web, Raleigh, NC (pp. 561–570). New York, NY: ACM. [Crossref], [Google Scholar]
23. Ophir, E., Nass, C., & Wagner, A. D. (2009). Cognitive control in media multitaskers. Proceedings of the National Academy of Sciences, 106(37), 15583–15587. doi:10.1073/pnas.0903620106 [Crossref], [PubMed], [Web of Science ®], [Google Scholar]
24. Otsu, N. (1979). A threshold selection method from gray-level histograms,”. IEEE Trans. Systems, Man, and Cybernetics, 9(1), 62–66. doi:10.1109/TSMC.1979.4310076 [Crossref], [Web of Science ®], [Google Scholar]
25. Smith, R. (2007, September). An overview of the Tesseract OCR engine. Proceedings of the ninth international conference on document analysis and recognition, Parana, Brazil (Vol. 2, pp. 629–633). IEEE. [Crossref], [Google Scholar]
26. Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., … Rabinovich, A. (2015). Going deeper with convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition, Boston, MA (pp. 1–9). IEEE. [Crossref], [Google Scholar]
27. Talukder, K. H., & Mallick, T. (2014). Connected component based approach for text extraction from color image. In Proceedings of the 17th IEEE International Conference on Computer and Information Technology, Helsinki, Finland (pp. 204–209). IEEE. [Crossref], [Google Scholar]
28. Tossell, C., Kortum, P., Rahmati, A., Shepard, C., & Zhong, L. (2012, May). Characterizing web use on smartphones. In Proceedings of the SIGCHI conference on human factors in computing systems, Austin, TX (pp. 2769–2778). ACM. [Crossref], [Google Scholar]
29. Wang, R., Chen, F., Chen, Z., Li, T., Harari, G., Tignor, S., … Campbell, A. T. (2014, September). StudentLife: Assessing mental health, academic performance and behavioral trends of college students using smartphones. Proceedings of the 2014 ACM international joint conference on pervasive and ubiquitous computing, Seattle, WA (pp. 3–14). ACM. [Crossref], [Google Scholar]
30. Yeykelis, L., Cummings, J. J., & Reeves, B. (2014). Multitasking on a single device: Arousal and the frequency, anticipation, and prediction of switching between media content on a computer. Journal of Communication, 64(1), 167–192. doi:10.1111/jcom.12070 [Crossref], [PubMed], [Web of Science ®], [Google Scholar]
31. Yeykelis, L., Cummings, J. J., & Reeves, B. (2018). The fragmentation of work, entertainment, E-Mail, and news on a personal computer: Motivational predictors of switching between media content. Media Psychology, 21(3), 377–402. doi:10.1080/15213269.2017.1406805 [Taylor & Francis Online], [Web of Science ®], [Google Scholar]
32. Chiatti, Agnese. "Information Extraction and Retrieval from Digital Screenshots–Archiving in situ Media Behavior." (2019).
33. Suatap, Chayanin, and Karn Patanukhom. "Development of Convolutional Neural Networks for Analyzing Game Icon and Screenshot Images." International Journal of Pattern Recognition and Artificial Intelligence 36.14 (2022): 2254023.
34. Muise, Daniel, et al. "Selectively localized: Temporal and visual structure of smartphone screen activity across media environments." Mobile Media & Communication 10.3 (2022): 487-509.
35. White, Ryen W. "Intelligent futures in task assistance." Communications of the ACM 65.11 (2022): 35-39.
36. Feiz, Shirin, et al. "Understanding Screen Relationships from Screenshots of Smartphone Applications." 27th International Conference on Intelligent User Interfaces. 2022.
37. Ohkawa, Yuki, and Takafumi Nakanishi. "Detection Method of User Behavior Transition on Computer." Advanced Data Mining and Applications: 18th International Conference, ADMA 2022, Brisbane, QLD, Australia, November 28–30, 2022, Proceedings, Part II. Cham: Springer Nature Switzerland, 2022.
38. Alahmadi, Mohammad, Abdulkarim Malkadi, and Sonia Haiduc. "UI screens identification and extraction from mobile programming screencasts." Proceedings of the 28th International Conference on Program Comprehension. 2020.
39. Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32.
40. Meyer, A. N., Satterfield, C., Züger, M., Kevic, K., Murphy, G. C., Zimmermann, T., & Fritz, T. (2020). Detecting developers’ task switches and types. IEEE Transactions on Software Engineering, 48(1), 225-240.
