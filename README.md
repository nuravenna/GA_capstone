# Capstone Project - Identifying and Classifying Extremist and Radicalizing Language


## 1. Problem Statement

Can a machine learning model successfully identify and classify online extremist content (defined as containing extreme speech, see definition below), filtering it out from superficially similar but definitively non-extremist language content?

I will build a machine learning model that will be able to:

- Classify extremist from non-extremist text content over large datasets (more that 35_000 observations)
- Have an accuracy score of at least 10% over the baseline (the majority class)
- Be able to discern non-extremist content from a relatively similar content class in the dataset (e.g. observations labeled from a non-extremist politics site that contains some intense languages and debate, as opposed to posts from a recipe website)


## 2. Definitions:

**Extremism** is defined by the Anti-Defamation League as “a concept used to describe religious, social or political belief systems that exist *substantially outside of belief systems more broadly accepted in society*. Extreme ideologies often seek radical change in the nature of government, religion or society.”

Not every extremist movement is inimical per se—- the Anti-Defamation League cites the abolitionist movement of the 19th century (extremist in its time) as an example of a "good" extremist movement, however, according to the ADL "most extremist movements exist outside of the mainstream because many of their views or tactics are objectionable".

**Extreme speech** is any language that contains and/or promotes extremism-- citing, encouraging, justifying or otherwise spreading ideas that are acutely outside the generally-accepted beliefs of the greater society to which the language users belong.

**Radicalization** is defined by the Center for Research on Extremism as “the gradual social process into extremism… often applied to explain changes in ideas or behavior.”

**Radicalizing language** is any language that can radicalize those who are exposed to it, usually but not necessarily over time.


## 3. Data:

Dataset source: Peeter, Stijn; Hagan, Sal; Das, Partha: “Salvaging the Internet Hate Machine: Using the discourse of extremist online subcultures to identify emergent extreme speech”, February 2020, presented at the 12th ACM Web Science Conference 2020 (https://zenodo.org/record/3676483#.YbAkv_HMLfF)

I was able to obtain scraped subreddit post data from a research project at the University of Amsterdam, cited just above, which was fortuitous because two of the subreddits in this corpus have since been banned by Reddit. The_Donald and ChapoTrapHouse are both no longer active, both banned in June 2020, roughly seven months after their subreddit content for the above project had been scraped by the researchers (between October 1st and November 1st of 2019). The two subreddits were closed due to their promoting violence and breaking Reddit content guidelines. The University of Amsterdam research project had used the subreddit post corpus as a test set for their "lexicon of 'extreme speech' (initially extracted from a corpus of 3_335_265 posts from 4chan's /pol/ sub-forum)...used to detect hate speech and extreme speech on online platforms".  

This corpus, which I repurposed for my classification project, was acquired in the form of a csv of 3_618_557 rows, each row containing the content of a different post as well as which subreddit the post came from. Each post originates from one of four subreddits:

- **r/politics:** Reddit's general politics subreddit, a forum for news and discussion about US politics, which, significantly, contains a great deal of topical content and surface language (i.e. passionate and intense political and cultural debates and other exchanges) similar to the subreddits in the corpus deemed 'extremist'-- ideal as the non-extremist control category. Currently active. 7.8 million members. Comprises 2_379_546 of the posts in this dataset, 66% of the corpus. Used as the non-extremist category for the classification model.
   
- **r/The_Donald:** A forum where participants posted comments in support of Donald Trump. Initially created in 2015 after he announced his presidential campaign. Currently inactive, banned by Reddit in June 2020 for violating Reddit harassment policies. 790_000 members at its height. Comprises 878_217 of the posts in this dataset, 24% of the corpus. Used as an extremist category for the classification model.
   
- **TheRedPill:** A forum promoting traditional gender roles, antifeminism, rape culture and "hegemonic masculinity" (https://en.wikipedia.org/wiki/Controversial_Reddit_communities). In quarantine since 2018. 149_432 members. Comprises 348_552 of the posts in this dataset, 9% of the corpus. Used as an extremist category for the classification model.
   
- **ChapoTrapHouse:** A forum of socialist memes and posts, described as left-leaning, anti-cop, pro-conspiracy theory. Currently inactive, banned by Reddit in June 2020 (at the same time it banned The_Donald, citing a crackdown on ‘pro-hate’ communities). 160_000 members at its height. Comprises 12_242 of the posts in this dataset, .03% of the corpus.


## 4. Methodology I: Determining what is and is not extremist content

I decided to use the subreddit the post originated from as a proxy for whether or not the post itself is extremist, therefore, for all intents, labeling entire subreddits as 'extremist' or 'non-extremist'. I am aware that many would object to an entire subreddit being defined as such, as opposed to labeling as 'extremist' the individual posts within it, ones that contain a sufficient concentration of extreme speech. The way to do the latter would be to build or extract a lexicon of content words that reflect extreme speech and compare each post to that lexicon, getting a measure of how many words overlap between the two, and using that measure above a certain cut-off to identify extremist posts within the corpus at large, regardless of which subreddit the post came from. Indeed, I plan for that to be an upcoming step of further research for this project, as several NLP and sentiment analysis tools exist that have that capacity.

However, as an initial phase of this project I decided that subreddits, especially consistently problematic subreddits such as TheRedPill and The_Donald, have enough of a unique linguistic footprint to be used as classification labels. This was after doing research on the ethos and subject of each subreddit, as well as its history with the Reddit platform. Reddit itself has banned two of the subreddits in the modeling dataset I classified as extremist (The_Donald and ChapoTrapHouse), and the third subreddit (TheRedPill) is in quarantine and therefore still accessible, but retains its notoriety as a forum that (to say the least) cultivates and condones opinions and language that are not shared by the society at large, and are therefore, by definition, 'extremist'.

I did not search out quarantined subreddits to define as extremist, but rather followed the lead of the original researchers who collected this dataset I used for my project-- they scraped post content from one general politics forum and three controversial forums that they then tested against the extreme speech lexicon they had developed. In their summary of their project on zenodo.org (see the link under the Data section, above) they admit that Reddit, and by extension the subreddits in their scraped dataset, can be described as mainstream when compared to 4chan and other anonymous internet forums. Relatively mainstream or not, there was a reason the researchers chose those particular subreddits for their project, namely that at least three of them could be expected to contain extreme speech to a sufficient degree that Peeter, et al could test their extremist language benchmark tool against the subreddit post content in question and detect classifiable patterns. 

In addition, after the subreddit posts were collected by the researchers, but before I downloaded and repurposed the data for this project, Reddit made the organizational decision to ban two of the subreddits used by the researchers, thus unintentionally backing up their conclusions. So, as far as the initial iteration of this extremist language classification project, I am satisfied that using the four subreddits in question as delineators of extremist vs nonextremist text is a serviceable, if not ideal, data science strategy.


## 5. Methodology II:

Multiple classification models were applied to a dataset comprised of content scraped from four different but overtly similar subreddits (see the discussion and descriptions, above). Each model was scored by the accuracy metric and a confusion matrix, displaying all correct classifications as well as type I and type II errors.

I used a balanced dataset from this corpus for the modeling: 40_000 posts total, 20_000 from the three subreddits deemed as extremist, 20_000 from the r/politics subreddit used as a non-extremist control classification category. Since the classes are balanced 50/50, accuracy is the preferred scoring metric.

NLP techniques were used to clean and preprocess the post data in preparation for being input to the various classification models. The final dataframe used as input for the modeling had 40_000 rows (each row a different scraped Reddit post) and eight columns:

- **body:** the original raw post text originally scraped from the Reddit site

- **subreddit:** the subreddit the post came from (politics, ChapoTrapHouse, TheRedPill, or The_Donald)

- **word_count:** the word count of the post

- **tokenized:** the post text after being run through an NLP preprocessing function, tokenizing and lower-casing the text, and removing punctuation

- **clean_content:** the post text after being run through an NLP stemming function, performing all of the processing in the tokenized column PLUS stemming each word in the post

- **extreme:** containing a binary marker noting whether the post for the row in question is deemed extremist (1) or non-extremist (0), based upon which subreddit the post came from (0 if from r/politics, 1 if from any of the other three subreddits)

- **adjectives:** a list of all the adjectives in the associated post

- **adj_string:** all the adjectives in the associated post as strings separated by a space

NLP preprocessing:

- Create a word count of each subreddit post
   
- Tokenize, stem, remove punctuation and special characters (the original, raw posts retained as well as a column of the tokenized posts and a column of the  stemmed posts, in order to compare performance in the models)
   
- Identify all the adjectives in each post using Spacy’s Large English Model, then aggregating that to a total count of adjectives for each subreddit

Features, parameters and transformations tested by the models:

- with/without Standard English stopword removal
   
- Word Count Vectorization (with various parameters)
   
- with/without TF-IDF transformation
   
- With/without StandardScaling
   
- Raw text vs Tokenized text vs Stemmed text
   
- with/without the word count of each post included as input for the observation

Models used:

- Logistic Regression 
   
- Support Vector Machine 
   
- Naive Bayes
   
- Decision Tree Classifier 
   
- Random Forest Classifier 
   
- Extra Trees Classifier


## 6. Conclusions and Recommendations

The two highest-scoring models were both Logistic Regression models. The SVM classifier performed well on the natural language data, however, when combined with the ideal hyperparameter of no (or very high) max features after CountVectorizer transformation, the SVM was unable to overcome the limits of computing power being run from a Jupyter Lab notebook on my machine, with the kernel shutting down in all but one instance of running it. The same went for the three tree classifier models I attempted to use (Decision Tree, Random Forest and Extra Trees). Therefore the bulk of the completed model data I have for this first iteration of this project comes from Logistic Regression.

(With max features set to None and run across the 40_000 count modeling dataset, the count vectorizer returns 33_272 columns of word tokens-- a very large number of features)

The hyperparameters that seem to give the highest scores are 1) unigrams only, 2) English stopwords removed, and 3) max features set to None (or at least the higher the better). Running the model across the tokenized text OR the raw text seems to improve the score, as opposed to using the processed (stemmed) text.

The best performing model is a Logistic Regression model with an accuracy score of .673. It was run across the count vectorized raw post text with no set max features, with English stopwords not removed and an ngram range of 1,1. This model has the best tradeoff between false positives and false negatives, at 1775 false positives (the lowest count for all the models) to 1471 false negatives.

The second best model is another Logistic Regression model, with an accuracy score of .672. It was also run across the count vectorized raw post text, this time with max features set to 16_500 (half of the total number of potential word count features for the dataset), with English stopwords removed, an ngram range of 1,1, and the post word count column added to the post text column as input for the model. This model had the second lowest count of false positives (after the best-performing model).

Considering how similar the extremist posts are to the non-extremist posts in this dataset, I am well-satisfied with an accuracy score of 67%.

In addition, since 67% is more than 10 points over the baseline accuracy score of 50% for this evenly balanced dataset, I have succeeded in my goals set for the Problem Statement.

I have concluded that it is possible to build a classification model that can tell extremist content from non-extremist content (as defined by the precepts of this project, outlined above in the Methodology I section). Going forward on the project, I need to run subsequent models across a computing platform with more power. This is because one of the consistently best-scoring input parameters for this Reddit post corpus is word count vectorization with no max feature limits, resulting in very large features matrices of up to 33_272. As it stands now, the higher the max features on the count vectorizer the better the scores, but the more likely the kernel will be shut down due to too much computational effort, a tradeoff that needs to be addressed in the next phase of the project.

So far the other two consistently best-performing parameters are unigrams and English stopword removal. Models run across either the raw or the tokenized text seem to have equally better accuracy scores than those run across the stemmed text, however models using the raw text seem to have a better balance of false positives/false negatives compared to those using the tokenized text. I am currently unsure of why that is the case, and hesitate to guess-- I need more information from subsequent iterations of the project (see below).

Since the subset of data I used for the modeling so far came from a larger dataset of more than three million observations, another motivation for using more computing power is the ability to potentially run a model across the entire dataset, likely resulting in very different outcomes and a great deal of additional information about patterns of extreme speech. Additional NLP processing using Spacy and other sophisticated platforms like HuggingFace may possibly result in a model that can make use of bigrams and trigrams in the data, instead of being limited to unigrams (currently, ngram ranges of 1,1 were almost unanimously selected by grid searches as the ideal range for the ngram parameter-- however using Named Entities and part of speech dependencies may allow me to massage the post text into more useful 2- and 3-unit ngrams, thus improving the model's classification abilities).

While I was disappointed that I did not get further into Spacy, let alone neural nets and HuggingFace, I am looking at this as phase 1 of an ongoing project, and I’ve now got a pretty robust roadmap for going forward.


## 7. Executive Summary

Extremism and radicalization exist across all languages and nations, but it has been on the rise in this country in recent years, along with hate crimes potentially triggered by its viewpoints. Hate groups have been steadily increasing in the US since the early aughts, with a spike in patriot groups, nativist extremist groups, and militias starting in 2008. In recent years, posts and other online interactions have increasingly contained language that can be defined as extremist (either politically or culturally), and that has been used to foment dissatisfaction, anger, and violence, thereby fitting the definition of radicalization. 

The study of the toxicity of online subcultures is an important research area from a societal standpoint. It is to our benefit to be able to hone in on and identify such language as it arises, whether to study it in order to determine its effect on those who create and consume it, to monitor its content to prevent associated outbreaks of violence in the real world, or, in extreme cases (from a free speech/censorship standpoint), to close the communication platform down completely, thus preventing any further exchanges. 

The problem with this area of research arises when one considers the massive amount of data to filter through on the platforms of interest (with more being created every second), as well as the whack-a-mole response of the language creators after being censured or shut down–- quickly re-establishing their profiles/content elsewhere on the web, at which point the entire process starts over. Assuming the above problem can be solved, any model that can be built to successfully identify extremist language in real time, especially while filtering it from non-extremist but otherwise similar language, would have very promising research implications and applications.

I have created a Logistic Regression model that can classify extremist content from non-extremist content with a 67% accuracy rate. Using the subreddit the post originated from as a proxy of whether or not it is an extremist post (a controversial decision, the reasons for which I expound upon in the Methodology I section, above), when applied to the natural language content of a post my model can discern whether or not it derives from a subreddit defined as extremist. The subsequent notebooks go over the various models applied to this problem, as well as the NLP preprocessing undergone by the data, an overview of the best-scoring hyperparameters, and a demo of the best-performing model.

I had a hard time initially finding data for this project–- the good news is that social media platforms are doing a good job of finding and shutting down extremist sites, the bad news is that such data being available for research is increasingly rare. I ended up finding a dataset of Reddit posts from three extremist subreddits that have since been closed down or quarantined by the Reddit platform: r/The_Donald, r/TheRedPill, and r/ChapoTrapHouse, interspersed with posts from the general r/politics subreddit that remains current on Reddit. These subreddit posts were collected by the initial researchers (see the Data section, above) across a one month period between October and November 2019. This dataset had initially been used by the researchers, a team from the University of Amsterdam, as a test set for an extreme speech lexicon they had developed, from a corpus scraped from the 4chan platform.

Rather than making the decision myself (and possibly being constrained by my biases), I left the determination of what was an extremist vs nonextremist subreddit up to the University of Amsterdam researchers, and Reddit itself. The original researchers had chosen which subreddits to collect in order to test for extremist language, and the results of their classification model, using their extreme speech lexicon, confirmed their initial judgements on which were and were not 'extremist' subreddits. The Reddit platform later provided vindication by banning two of the subreddits in question just seven months after the study (the third extremist subreddit, TheRedPill, remains accessible but has been in quarantine since 2018).

The data: 
- The dataset I used was a randomly sampled subset of 40_000 observations from the larger (3.5 million) dataset of subreddit posts scraped from Reddit
- The classes were equal, 50% extremist subreddits (from The_Donald, TheRedPill and ChapoTrapHouse, 20_000 total) and 50% non-extremist subreddits (from the r/politics subreddit, 20_000 total), so any successful model would have to have an accuracy of more than 50%

The best-performing model:
- The best performing model was a Logistic Regression model, run across the raw text after count vectorization (unigrams only, no maximum features, no stopword removal)
- This model had an accuracy score of 67%, with a recall score of 70%, a precision score of 66% and an F1 score of 68%

Conclusion:
It is possible to create a model to classify extremist from non-extremist text, even in the case where there are some topical overlaps between the observations labeled extremist vs non-extremist. This is good news, because the authors of extremist language are getting increasingly canny and sophisticated about creating new content, both from a logistical standpoint (e.g, creating new subreddits, social media profiles, etc, as old ones are discovered and shut down) as well as from a linguistic standpoint (e.g. creating slang, masked and coded language, obscure vernacular and other rapid lexical innovation to encode and disperse their beliefs).

While 67% accuracy does not seem that impressive, considering how similar the non-extremist posts were to the extremist posts (with a great deal of lexical and semantic overlap) this is quite a good return.

Subsequent steps to take in this project:

- Use the Spacy Large English Model as part of a classification pipeline, e.g. run over only the adjectives, and/or the Named Entities, and/or the Subjects of each post
- Use a one-dimensional CNN model, and other neural network models
- Try HuggingFace transfer models
- Get more computing power
- Acquire different input data for modeling, including a dataset comprised of language that is incontestably 'extremist' and incontestably ‘non-extremist’


## 8. References

1) “Salvaging the Internet Hate Machine: Using the discourse of extremist online subcultures to identify emergent extreme speech”, Peeter, Stijn; Hagan, Sal; Das, Partha, February 2020, presented at the 12th ACM Web Science Conference 2020 
https://zenodo.org/record/3676483#.YbAkv_HMLfF


2) Controversial Reddit communities-- Wikipedia
https://en.wikipedia.org/wiki/Controversial_Reddit_communities


3) "Twitch, Reddit crack down on Trump-linked content as industry faces reckoning", Politico, 6/29/2020 
https://www.politico.com/news/2020/06/29/reddit-bans-pro-trump-forum-in-crackdown-on-hate-speech-344698

4) "Reddit Banned A Ton Of Subreddits Including r/The_Donald And r/ChapoTrapHouse", BuzzFeed News, 6/29/2020 
https://www.buzzfeednews.com/article/juliareinstein/reddit-bans-subreddits-thedonald-chapotraphouse


5) "Male Supremacy", Southern Poverty Law Center
https://www.splcenter.org/fighting-hate/extremist-files/ideology/male-supremacy


6) "What is r/TheRedPill, the infamous men’s rights subreddit?", DailyDot, 5/21/2021
https://www.dailydot.com/debug/reddit-red-pill/


7) "Spitting out the Red Pill: Former misogynists reveal how they were radicalised online", The New Statesman, 9/9/2021
https://www.newstatesman.com/science-tech/2017/02/reddit-the-red-pill-interview-how-misogyny-spreads-online


8) "Reddit ‘Quarantines’ White Nationalist Subreddits", Daily Beast, 9/29/2018
https://www.thedailybeast.com/reddit-quarantines-white-nationalist-subreddits


9) “The rise of domestic extremism in America”, Washington Post, 4/12/2021
https://www.washingtonpost.com/investigations/interactive/2021/domestic-terrorism-data/


10) “The rise of domestic terrorism is fueled mostly by far-right extresmists", The Philadelphia Inquirer, 4/12/2021
https://www.inquirer.com/news/nation-world/domestic-terrorism-analysis-study-far-right-20210412.html


11) “Were the Sikh Temple Killings Preventable?”, Mother Jones, 8/9/2012
https://www.motherjones.com/politics/2012/08/sikh-temple-killings-preventable-homeland-security/


12) “Radicalization and Violent Extremism: Lessons Learned from Canada, the UK and the US”, National Institute of Justice Report, July 2015
https://www.ojp.gov/pdffiles1/nij/249947.pdf


13) “What is radicalization?”, Center for Research on Extremism
https://www.sv.uio.no/c-rex/english/groups/compendium/what-is-radicalization.html

