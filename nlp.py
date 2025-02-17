'''

**********************************************************************************************************
                        🧬_NLP Terms, Text Preprocessing, Representing Text, and NLP Tasks_🧬
**********************************************************************************************************


🧬  In the previous chapter, we explored what Generative AI and Natural Language Processing (NLP) are, 
    their significance, and key technologies shaping the field.



🧬  Now, as you progress in understanding NLP, it’s crucial to learn how to prepare and represent text 
    data in a way that machines can understand. This chapter introduces the foundational steps 
    of text preprocessing, different methods for representing text as data, and some basic NLP 
    tasks that you can begin experimenting with.


**********************************************************************************************************
                                                🧬_Document_🧬
**********************************************************************************************************


🧬  Objective:

    By the end of this section, you should have a better understanding of the basic terms used in NLP.



🧬  What is Document?

    A document is a single piece of text, which can be anything from a single sentence to an entire book. It is the basic unit of text that NLP models process.

    Documents can be diverse in nature, such as emails, web pages, articles, or tweets.



🧬  Example:
    · A single news article from a newspaper.
    · A tweet: “Just watched an amazing movie!”
    · An email: “Dear John, I hope this email finds you well…”


**********************************************************************************************************
🧬__🧬
**********************************************************************************************************


🧬  A corpus (plural: corpora) is a large collection of documents. It serves as the dataset on which 
    NLP models are trained and evaluated. A corpus typically contains documents that are related 
    by topic, language, or genre and is used to analyze linguistic patterns and build statistical models.



🧬  Example:
    A collection of all articles from a specific newspaper over a year.
    A dataset of customer reviews from an e-commerce website.
    The Gutenberg Corpus: A collection of literary texts from Project Gutenberg.




🧬  Feature


🧬  A feature is a measurable property or characteristic of the text that is used in machine learning  
    models.



🧬  Features are extracted from documents and can represent various aspects of the text, 
    such as the presence of specific words, the length of sentences, or the occurrence of particular 
    patterns.


**********************************************************************************************************
                                        🧬_Text Preprocessing_🧬
**********************************************************************************************************


🧬  Objective:

    By the end of this section, you should be able to explain what text preprocessing is, 
    why it’s important, and how to apply common preprocessing techniques to raw textual data.

    
**********************************************************************************************************
                                    🧬_Definition and Overview_🧬
**********************************************************************************************************


🧬  What is Text Preprocessing?

    In text processing or preprocessing, we apply a series of steps and transformations to raw text to make 
    it more suitable for analysis, modeling, and other NLP tasks.

    Whay is processing needed? Well, natural language is full of complexities - misspelling, diverse writing
    styles, varying puctuation - wich can hinder the performance of NLP models.

    That's why a given text first has to be streamlined into cleaner, standardized format, before it can be
    processsed by a language model.

    

🧬  Importance of Preprocessing in NLP Workflows:

    Improving Model Accuracy:
    Models trained on clean, consistent input data gernerally yield better accuracy and more robust predictions.


    Reducing Noise:
    Removing unecessary elements such as non-informative words (e.g., "the", "is", "and") or punctuation 
    simplifies patterns in the data, making patterns clearer.

    
    Efficiency:
    Preprocessing makes the data more compact and can reduce computational cost, enabling models to train and 
    predict faster.

    
    Generalization:
    Well-preprocessed data is easier for models to understand, enabling them to generalize across different
    datasets and tasks.



**********************************************************************************************************
                                            🧬_Core Techniques_🧬
**********************************************************************************************************


🧬  Tokenization: Splitting Sentences into Words


    Definition:
    Tokenization is the process of splitting text into smaller units called 'tokens'. Tokens can be whole
    words, but they also can be subwords or characters.


    Example:
    The sentence "Hello world!" can be tokenized into the following words: ["Hello", "world", "!"].


    Benefits:
    Allows models and algorithms to work on meaningful units (words or subwords) rather than entire sentences.




🧬  Stop word Removal, Lowercasing, and Punctuation Removal


    Stop word Removal: Stop words (e.g., "the", "and", "to") are common words that don't usually contribute
    much to the meaning of text. Removing them can reduce noise.


    Lowercasing:
    Converting all letters to lowercase ensures that words like "Apple" and "apple" are treated the same.


    Punctuation Removal:
    Removing punctuation can simplify text analysis, although in some tasks punctuation might carry meaning
    (like sentiment from exclamation marks!). The decision depends on the application.




🧬  Stemming and Lemmatization: Reducing Words to Their Root Forms


    Stemming:
    A heuristic method that chops off word endings to reduce a word to a base form (stem). For example, 
    "playing", "played", and "plays" would all be reduced to "play". Stemming can be crude and may not 
    always produce valid words.


    Lemmatization:
    A more intelligent approach that reduces words to their dictionary base form (lemma). It considers
    the part-of-speech and ensures the reduced form is an actual word. For example, "better" may lemmatize
    to "good" if you consider it as an adjective.


    
🧬  Introduction to Regular Expressions (Regex) for Text Manipulation


    Rexex Basics:
    Regular expressions are a mini-language used to find, match, and manipulate specific text patterns. 
    For instance, you can use a regex to identify all sequences of digits in a text and remove them.


    Common Uses:
    Cleaning out URLs, removing special characters, extracting certain word patterns, or validating 
    formats like email addresses.

    
**********************************************************************************************************
                    🧬_Applications: Preparing Datasets for Machine Learning Models_🧬
**********************************************************************************************************


🧬  The ultimate goal of text preprocessing is to transform raw data into a format that can be fed into
    machine learning or deep learning models. 

    This might involve


    >>> Building a pipeline:
    A sequence of steps that starts from raw text, apllies tokenization, remove stop words, apply stemming/
    lemmatization, and finally produce a list of clean tokens.


    >>> Vectorizing text:
    Converting text into numerical representations that models can understand. This can be done using
    techniques like Bag of Words, TF-IDF, or Word Embeddings.


    >>> Training and Evaluating Models:
    Once the data is preprocessed and vectorized, it can be used to train and evaluate models for tasks
    like sentiment analysis, text classification, or machine translation.


    >>> Fine-tuning and Iterating:
    The preprocessing steps and vectorization techniques can be fine-tuned based on model performance,
    allowing for iterative improvements in the NLP pipeline.


    >>> Feature Engineering:
    Creating consistent, clean textual inputs to ensure that models like a sentiment classifier or a text
    categorization system can learn patterns effectively.


**********************************************************************************************************
                                        🧬_Representing Text as Data_🧬
**********************************************************************************************************


🧬  Objective:
    By the end of this section, you should understand different approaches to representing text numerically
    so that machines can process it. You will gain insights into simple methods like Bag of Words as well as
    more advanced techniques like embeddings.

    
    __matrix-blue.gif__
🧬  https://colonycheese-natashaexpand.codio.io/.guides/img/matrix-blue.gif



🧬  What does "Representing Text as Data" Mean???

    Computers understand numbers, not words. To use text in machine learning models, we need to convert it 
    into a numverical form.

    Each piece of text - whether a word, a senctence, or an entire document - must be translated into a 
    format a machine can process, such as vectors (lists of numbers).


🧬  Challenges with Text Representation

    >>> Context and Meaning:
    Words have multiple meanings depending on context. For example, "bank" could mean a financial institution
    or the side of a river.

    
    >>> Dimensionality:
    Text datasets can be huge, and representing them often leads to very high-dimansional vectors (thousands 
    or millions of features).
    

    >>> Sparsity:
    Text data is often sparse, meaning most elements in the vector are zeros. This can make it challenging
    to process and analyze.


    >>> Semantics:
    Representing the meaning of words and sentences in a way that captures their relationships is a complex
    task.


🧬  Common Text Representation Techniques

    >>> Bag of Words (BoW):
    A simple method that represents text as a matrix of word counts. It ignores word order and context,
    focusing on word frequency.


    >>> Term Frequency-Inverse Document Frequency (TF-IDF):
    A more advanced technique that considers the importance of words in a document relative to their
    frequency across all documents.


    >>> Word Embeddings:
    Dense, low-dimensional vectors that capture semantic relationships between words. Word embeddings are
    learned from large text corpora using neural networks.


    >>> Character Embeddings:
    Similar to word embeddings, but they operate at the character level. They can capture morphological
    and syntactic information.


    >>> Contextual Embeddings:
    Embeddings that consider the context of words in a sentence. They are generated using models like BERT
    and GPT-3.


    >>> Document Embeddings:
    Vectors that represent entire documents, capturing their semantic content. They are useful for tasks
    like document classification and clustering.


    >>> Graph-Based Representations:
    Representing text as graphs to capture relationships between words or sentences. Graph-based methods
    can model complex dependencies in text data.


    >>> Hybrid Approaches:
    Combining multiple text representation techniques to leverage the strengths of each. For example,
    combining BoW with word embeddings to capture both word frequency and semantics.


    >>> Semantic Hashing:
    Mapping text to binary codes that capture semantic similarity. Semantic hashing is useful for
    information retrieval and similarity search.


    >>> Knowledge Graphs:
    Representing text as structured knowledge graphs to capture relationships between entities and concepts.
    Knowledge graphs are useful for question answering and information extraction tasks.


    >>> Attention Mechanisms:
    Techniques that learn to focus on specific parts of text data. Attention mechanisms are useful for
    tasks like machine translation and text summarization.


    >>> Transfer Learning:
    Leveraging pre-trained models to extract features from text data. Transfer learning can boost performance
    on downstream NLP tasks.


    >>> Multimodal Representations:
    Combining text with other modalities like images, audio, or video. Multimodal representations are useful
    for tasks like image captioning and video summarization.


    >>> Explainable Representations:
    Representations that can be interpreted by humans to understand how models make predictions. Explainable
    representations are crucial for building trust in AI systems.


    >>> Dynamic Representations:
    Representations that change over time or adapt to new data. Dynamic representations are useful for
    capturing evolving trends in text data.


    >>> Privacy-Preserving Representations:
    Techniques that protect sensitive information in text data. Privacy-preserving representations are
    crucial for handling confidential or personal data.


    >>> Multilingual Representations:
    Representations that capture text in multiple languages. Multilingual representations are useful for
    tasks like machine translation and cross-lingual information retrieval.


    >>> Domain-Specific Representations:
    Representations that capture specialized terminology and concepts in specific domains. Domain-specific
    representations are useful for tasks like scientific text analysis and medical document processing.


    >>> Scalable Representations:
    Techniques that can handle large text datasets efficiently. Scalable representations are crucial for
    processing big data and real-time text streams.


    >>> Robust Representations:
    Representations that are resilient to noise and errors in text data. Robust representations are crucial
    for handling noisy or incomplete text data.


    >>> Interpretable Representations:
    Representations that can be easily understood and analyzed by humans. Interpretable representations are
    crucial for explaining model predictions and building trust in AI systems.


    >>> Adaptive Representations:
    Representations that can adapt to changing contexts or tasks. Adaptive representations are useful for
    handling dynamic text data and evolving NLP tasks.


    >>> Fair Representations:
    Representations that are unbiased and equitable across different groups. Fair representations are crucial
    for building inclusive and ethical AI systems.


    >>> Low-Resource Representations:
    Techniques that can handle text data with limited labeled examples or resources. Low-resource representations
    are crucial for developing NLP models in resource-constrained settings.


    >>> Multiview Representations:
    Representations that combine multiple perspectives or sources of information. Multiview representations are
    useful for capturing diverse aspects of text data and improving model performance.


    >>> Adversarial Representations:
    Techniques that can defend against adversarial attacks on text data. Adversarial representations are crucial
    for building secure and robust NLP models.


    >>> Self-Supervised Representations:
    Techniques that learn representations from unlabeled text data. Self-supervised representations are useful
    for pretraining models and improving performance on downstream NLP tasks.


    >>> Zero-Shot Representations:
    Techniques that can generalize to new tasks or languages without additional training. Zero-shot representations
    are useful for adapting models to novel scenarios and domains.


    >>> Few-Shot Representations:
    Techniques that can learn from a small number of labeled examples. Few-shot representations are useful for
    developing NLP models in low-resource settings.


    >>> Multitask Representations:
    Representations that can handle multiple NLP tasks simultaneously. Multitask representations are useful for
    improving model efficiency and generalization.


    >>> Cross-Modal Representations:
    Representations that capture relationships between text and other modalities like images or audio. Cross-modal
    representations are useful for tasks like multimodal sentiment analysis and emotion recognition.


    >>> Hierarchical Representations:
    Representations that capture hierarchical structures in text data. Hierarchical representations are useful for
    modeling complex relationships between words, sentences, and documents.


    >>> Temporal Representations:
    Representations that capture the temporal dynamics of text data. Temporal representations are useful for


    >>> Sequential Representations:
    Representations that capture the sequential nature of text data. Sequential representations are useful for
    tasks like text generation and sequence labeling.


    >>> Spatial Representations:
    Representations that capture the spatial relationships between words or entities in text data. Spatial
    representations are useful for tasks like document layout analysis and information extraction.


    >>> Compositional Representations:
    Representations that can be combined to form more complex structures. Compositional representations are
    useful for modeling relationships between words and entities in text data.


    >>> Multiscale Representations:
    Representations that capture information at multiple levels of granularity. Multiscale representations are
    useful for modeling text data with varying degrees of detail.


    >>> Multiresolution Representations:
    Representations that capture information at multiple levels of resolution. Multiresolution representations
    are useful for modeling text data with varying levels of detail.


    >>> Multitask Learning:
    Learning multiple tasks simultaneously to improve model performance. Multitask learning is useful for
    leveraging shared information across tasks and domains.


    >>> Transfer Learning:
    Leveraging knowledge from pre-trained models to improve performance on new tasks. Transfer learning is
    useful for adapting models to different datasets and domains.


    >>> Meta-Learning:
    Learning to learn from a small number of examples. Meta-learning is useful for developing models that can
    quickly adapt to new tasks and environments.


    >>> Reinforcement Learning:
    Learning to make decisions by interacting with an environment. Reinforcement learning is useful for
    developing models that can learn from feedback and improve over time.


    >>> Unsupervised Learning:
    Learning patterns and structures from unlabeled data. Unsupervised learning is useful for discovering
    hidden relationships in text data and generating novel insights.


    >>> Supervised Learning:
    Learning to make predictions from labeled data. Supervised learning is useful for developing models that
    can classify text, extract information, or generate responses.


    >>> Semisupervised Learning:
    Learning from a combination of labeled and unlabeled data. Semisupervised learning is useful for developing
    models in scenarios where labeled data is limited or expensive to obtain.


    >>> Self-Supervised Learning:
    Learning from data without human annotations. Self-supervised learning is useful for pretraining models
    on large text corpora and improving performance on downstream tasks.


    >>> Multimodal Learning:
    Learning from multiple modalities like text, images, and audio. Multimodal learning is useful for developing
    models that can understand and generate content across different domains.


    >>> Multitask Learning:
    Learning multiple tasks simultaneously to improve model performance. Multitask learning is useful for
    leveraging shared information across tasks and domains.


    >>> Sematic Relationships:
    Similar words should have similar numeric representations. Capturing this isn't straightforward.


**********************************************************************************************************
                                        🧬_Bag of Words (BoW)_🧬
**********************************************************************************************************


🧬  Definition:
    BoW represents text by counting how many times each word appears. It ignores grammar, order, and context.


🧬  Process:
    Suppose you have a vocabulary of the top 5000 words. Each document is turnd into a vector of length 5000,
    where each position counts how often a certain word appears in that document. Each individual word is 
    a feature and the corrensponding count is the feature's value.


🧬  Pros:
    Simple, easy to implement


🧬  Cons:
    Loses context and meaning; "amazing movie" and "movie amazing" become identical vectors.


🧬  Example:
    Document: "I love NLP"

    Features: {"I": 1, "love": 1, "NLP": 1}

    
**********************************************************************************************************
                            🧬_Term Frequency-Inverse Document Frequency (TF-IDF)_🧬
**********************************************************************************************************


🧬  Definition:
    TF-IDF is an improvement over BoW. It considers not just how often a word appears in a single document
    (Term Frequency) but also how rare it is across all documents (Inverse Document Frequency).


🧬  Resuilt:
    Words that appear frequently in one document but are rare across the whole corpus get higher scores, 
    providing more discriminative power.


🧬  Pros:
    More informative than BoW, reduces the weight of commonly used words that carry less meaning.


🧬  Cons:
    Still doesn't capture word order or context well.


🧬  Example:

    >>> Documents: 
    "The dog barked.",
    "The cat sat on the mat.",
    "The cat sat on the mat."

    >>> Features:
    ['barked' 'bed' 'cat' 'dog' 'mat' 'on' 'sat' 'the']]


    >>> TF-IDF Matrix:
    
    [[0.         0.         0.37420726 0.         0.49203758 0.37420726
    0.37420726 0.58121064]
    [0.         0.49203758 0.37420726 0.         0.         0.37420726
    0.37420726 0.58121064]
    [0.65249088 0.         0.         0.65249088 0.         0.
    0.         0.38537163]]



**********************************************************************************************************
                                            🧬_N-grams_🧬
**********************************************************************************************************


🧬  Definition:
    Contiguous sequence of n items (typically words or characters) from a given text or speech.


🧬  Types:

    >>> Unigrams: Single words, e.g., "apple", "banana", "cherry".
    >>> Bigrams: Pairs of words, e.g., "apple pie", "banana split".
    >>> Trigrams: Triplets of words, e.g., "apple pie recipe", "banana split sundae".
    >>> N-grams: Sequences of n words or characters, e.g., "apple pie recipe with ice cream".
    

**********************************************************************************************************
                            🧬_Word Embeddings (e.g., Word2Vec, GloVe)_🧬
**********************************************************************************************************


🧬  Definition:
    Word embeddings map words into a lower-dimensional vector pace where similar words are placed closer
    together.


🧬  Benefits:
    Captures semantic relationships: "king" and "queen" might have similar vectors.


🧬  Reduces dimensionality:
    Instead of thousands of features, embeddings might have 100-300 dimensions.


🧬  Examples:

    >>> Word2Vec:
    Learns embeddings by predicting words from surrounding context or vice versa.

    >>> GloVe:
    Uses global word co-occurence statistics to produce word embeddings.


🧬  Pros:
    Richer representation of meaning, capturing synonyms and analogies.


🧬  Cons:
    Embeddings trained on one data set might not reflect the nuances of another domain unless further 
    fine-tuned.

    
**********************************************************************************************************
                                            🧬_Basic NLP Tasks_🧬
**********************************************************************************************************


🧬  Objective:
    By the end of this section, you should be familiar with some fundamental NLP tasks, their significance, 
    and how to apply basic tequniques or pre-trained tools to implement them.


    Definition and Overview:

    >>> Key Tasks in NLP:

        >>> Sentiment Analysis
        >>> Named Entity Recognition
        >>> Part-of-Speech (POS) Tagging
        >>> Text Classification

    

**********************************************************************************************************
                                            🧬_Sentiment Analysis_🧬
**********************************************************************************************************


🧬  What is Sentiment Analysis?

    Sentiment analysis aims to determine the emotional tone expressed in the text. It categorizes text as
    positive, negative, or neutral.

    Sentiment analysis is widely used in analyzing customer feedback, movie reviews, social media posts, 
    and more.


🧬  Example:

    >>> Input: "I absolutely love this phone! the battery life is amazing, and the camera takes stunning
    photos."

    >>> Output: Positive sentiment

    


**********************************************************************************************************
                                    🧬_Named Entity Recognition (NER)_🧬
**********************************************************************************************************


🧬  What is NER?

    Named Entity Recognition (NER) identifies and classifies entities in a text into predefined categories
    such as person names, localizations, organizations, dates, and more. It helps to extract structured
    information from unstructured text.


🧬  Example:

    >>> Input: "Tony Kross visited Athens in August 2012."

    >>> Output:

        >>> Person: Tony Kross
        >>> Location: Athens
        >>> Date: August 2012


**********************************************************************************************************
                                    🧬_Part of Speech (POS) Tagging_🧬
**********************************************************************************************************


🧬  What is part of speech tagging (POS)

    POS tagging involves labeling each word in a sentence with its grammatical part-of-speech, such as 
    noun, verb, adjective, etc. This helps in understanding sentence structure, wich is useful for parsing,
    information extraction, and subsequent NLP tasks.


🧬  Example (pseudocode)

    >>> Input: "The quick brown fox jumps over the lazy dog."

    >>> Output:

        >>> The: Determiner
        >>> quick: Adjective
        >>> brown: Adjective
        >>> fox: Noun
        >>> jumps: Verb
        >>> over: Preposition
        >>> the: Determiner
        >>> lazy: Adjective
        >>> dog: Noun



**********************************************************************************************************
                                            🧬_Text Classification_🧬
**********************************************************************************************************


🧬  What is Text Classification

    Text classification assigns categories or labels to documents, sentences, or phrases based on their 
    content. Examples include classifying emails into "spam" or "not spam", categorizing news articles by
    topic, or labeling product reviews by sentiment or genre.


🧬  Example:

    >>> Input: "Breaking News: Stock prices plunge as markets react to global uncertainty."

    >>> Output: "Finance" or "Business"


🧬  Code example:

'''
text = "Breaking News: Stock prices plunge as markets react."
if "stock" in text.lower() or "markets" in text.lower():
    print("Category: Finance")
else:
    print("Category: Other")
'''


🧬  Summary and significance

    >>> These tasks serve as building blocks for more complex applications (e.g., chatbots, recommendation 
        systems, information extraction tools).

    >>> They enhance search engines, improve human-computer interaction, and provide insights into unstructured
        data.

        
**********************************************************************************************************
                                        🧬_Techniques and Tools_🧬
**********************************************************************************************************


🧬  Pretrained Models and Libraries:

    >>> SpaCy:
        A fast, industrial-strength NLP library with pretrained models for tokenization, POS tagging, NER, 
        and more.

    >>> NLTK (Natural Language Toolkit):
        One of the earliest and most widely used NLP libraries for beginners. It provides tools for tokenization,
        stemming, lemmatization, and more.

    >>> Gensim:
        A library for topic modeling, document similarity, and word embeddings like Word2Vec.

    >>> Transformers (Hugging Face):
        A library that provides access to pretrained models like BERT, GPT-3, and T5 for various NLP tasks.

    >>> Scikit-learn:
        A popular machine learning library that provides tools for text classification, clustering, and more.

    >>> TensorFlow and PyTorch:
        Deep learning frameworks that offer tools for building and training custom NLP models.


🧬  Practical Steps:

    >>> Load a pretrained model from SpaCy or Hugging Face.
    >>> Tokenize text into words or sentences.
    >>> Apply POS tagging, NER, or sentiment analysis to the text.
    >>> Use the output for downstream tasks like text classification or clustering.

        >>> For sentiment analysis, you might start with a BoW or TF-IDF representation, and then train 
            a classifier (like logistic regression or a neural network) on labeled data, to predict sentiment.

        >>> For NER, tools like SpaCy come with pretrained models that can identify common entity types 
            out-of-the-box.

        >>> For POS tagging, libraries like NLTK or SpaCy can annotate each word in a sentence with its 
            part-of-speech, helping you understand sentence structure.

        >>> For text classification, you can start simple with a BoW representation and a machine learning
            model and then move to embeddings for better performance.




🧬   In Summary:

    >>> Text preprocessing involves cleaning and preparing raw text, including tokenization, removing noice
        (like stop words and punctuation), and normalizing words with stemming or lemmatization.

    >>> Representing text as data transforms text into numeric vectors that models can understand, ranging 
        from simple counts (BoW) to more sophisticated embeddings (Word2Vec, GloVe) that capbure semantic
        meaning.

    >>> Basic NLP tasks like sentiment analysis, NER, POS tagging, and text classification are practical
        ways to gain experience and see tangible results in your early NLP projects.


As you continue learning, these fundamental concepts will enable you to tackle more advanced topics
and state-of-the-art NLP techniques with confidence.


**********************************************************************************************************
                                    🧬_Intruduction to LLMs_🧬
**********************************************************************************************************


🧬   Welcome to the world of Large Language Models (LLMs)!

    These models have recently gained significant attention for their remarkable ability to understand
    and generate human-like text across a wide range of tasks.

    This lesson covers what language models are, how they evolved, what makes transformers and LLMs special,
    and provides a hands on introduction to using them.


🧬  Objective:

    By the end of this lesson, you should be able to:

    >>> Understand what a language model is and how it has evolved from simple statistical models to 
        advanced neuronal architectures.

    >>> Recognize the significance of transformaers and the attention mechanism.

    >>> Understand the concepts of pre-training, fine-tuning, tokens, embeddings and how LLMs generate text.

    >>> Identify popular LLMS like GPT-3, BERT, and T5 and their applications in NLP tasks and their 
        unique characteristics.

    Ready? Let's dive in!

    
**********************************************************************************************************
                                        🧬_What is a language model_🧬
**********************************************************************************************************


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


**********************************************************************************************************
🧬__🧬
**********************************************************************************************************


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


**********************************************************************************************************
🧬__🧬
**********************************************************************************************************


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬




**********************************************************************************************************
🧬__🧬
**********************************************************************************************************


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬




**********************************************************************************************************
🧬__🧬
**********************************************************************************************************


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬




**********************************************************************************************************
🧬__🧬
**********************************************************************************************************


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬




**********************************************************************************************************
🧬__🧬
**********************************************************************************************************


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬




**********************************************************************************************************
🧬__🧬
**********************************************************************************************************


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬




**********************************************************************************************************
🧬__🧬
**********************************************************************************************************


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬




**********************************************************************************************************
🧬__🧬
**********************************************************************************************************


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬




**********************************************************************************************************
🧬__🧬
**********************************************************************************************************


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬




**********************************************************************************************************
🧬__🧬
**********************************************************************************************************


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬




**********************************************************************************************************
🧬__🧬
**********************************************************************************************************


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬




**********************************************************************************************************
🧬__🧬
**********************************************************************************************************


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬




**********************************************************************************************************
🧬__🧬
**********************************************************************************************************


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬




**********************************************************************************************************
🧬__🧬
**********************************************************************************************************


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬




**********************************************************************************************************
🧬__🧬
**********************************************************************************************************


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬




**********************************************************************************************************
🧬__🧬
**********************************************************************************************************


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬




**********************************************************************************************************
🧬__🧬
**********************************************************************************************************


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬




**********************************************************************************************************
🧬__🧬
**********************************************************************************************************


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬




**********************************************************************************************************
🧬__🧬
**********************************************************************************************************


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬




**********************************************************************************************************
🧬__🧬
**********************************************************************************************************


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬




**********************************************************************************************************
🧬__🧬
**********************************************************************************************************


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬




**********************************************************************************************************
🧬__🧬
**********************************************************************************************************


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬




**********************************************************************************************************
🧬__🧬
**********************************************************************************************************


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬




**********************************************************************************************************
🧬__🧬
**********************************************************************************************************


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬




**********************************************************************************************************
🧬__🧬
**********************************************************************************************************


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬




**********************************************************************************************************
🧬__🧬
**********************************************************************************************************


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬




**********************************************************************************************************
🧬__🧬
**********************************************************************************************************


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬




**********************************************************************************************************
🧬__🧬
**********************************************************************************************************


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬




**********************************************************************************************************
🧬__🧬
**********************************************************************************************************


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬




**********************************************************************************************************
🧬__🧬
**********************************************************************************************************


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬




**********************************************************************************************************
🧬__🧬
**********************************************************************************************************


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬




**********************************************************************************************************
🧬__🧬
**********************************************************************************************************


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬




**********************************************************************************************************
🧬__🧬
**********************************************************************************************************


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬




**********************************************************************************************************
🧬__🧬
**********************************************************************************************************


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬


🧬




**********************************************************************************************************
🧬__🧬
**********************************************************************************************************
'''