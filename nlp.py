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


🧬  In data science, we speak of models as computational representations of real-world systems. For example,
    a model might predict the weather, recommend movies, or classify images.


🧬  A language model is a type of model that learns the probability distribution of sequences of words. 
    It is trained on a large dataset of text to determine how likely a particular sequence of words is to 
    occur. This involves calculating probabilities for word combinations based on patterns in the data.


🧬  Why sequences of words?

    Language is inherently sequential: not only are words spoken or written one after another, but they are 
    also understood in this order.

        >>> DOG BITES MAN
        (the order of words affects the meaning of the sentence)

    The meaning and correctness of a sentence often depend on the order of words. A language model considers
    this order to predict or generate text.

    For example, in English, the phrase "the cat" is more likely than "cat the". The language model assigns 
    a higher probability to the former sequence and uses this information to understand and generate coherent 
    text.


🧬  large-language-models-2-1: How do large language models “learn”?

    A. By being programmed with a set of rules for generating text.
    B. By copying the text it has seen during training.
🧬  C. By being trained on a large amount of text data.
    D. By using pre-defined templates to generate text.


✔️ Correct! Large language models learn by being trained on a large amount of text data.



🧬  A simple example¶

    Simply put, a language model just tries to predict the next word in a sentence given the previous words.

    >>> For example, if you've seen the phrase:

    "Once upon a_______"

    >>> a language model might predict "time" as the next word because that's a common continuation.

    >>> You might recognize this from teh way auto-complete works. Modern chatbots are just more advanced versions!


🧬  large-language-models-2-2: Which of the following best describes a language model?

🧬  A. A model that learns the probability distribution of sequences of words.
    B. A model that learns the probability distribution of individual words.
    C. A model that generates random sequences of words.
    D. A model that translates text between languages.

✔️ Correct! A language model learns the probability distribution of sequences of words.


**********************************************************************************************************
                                🧬_🧠 What is a large language model?_🧬
**********************************************************************************************************


🧬  Now you know what a language model is. But what is a Large Language Model?

    A Large Language Model (LLM) is a type of artificial intelligence algorithm that uses deep leaning 
    techniques and massive data sets to achieve general-purpose language understanding and generation.
    So instead of requiring a separate model for translating text, summarizing articles, or answering 
    questions, LLMs can be used for all these tasks and more.

    LLMS are pre-trained on vast amounts of data, often including sources like the 'Common Crawl' dataset and 
    'Wikipedia'. This pre-training allows LLMs to learn the structure and patterns of language, making them 
    capable of understanding and generating text in a variety of contexts.

    LLMs are designed to recognize, summarize, translate, predict and generate text and other forms of content
    based on the knowledge gained from their training.


🧬  What's new?

    What makes modern LLMs so powerful? Some key innovations include:

        >>> Transformaer Model Architecture:
            LLMs are based on the transformer model architecture, which allows them to process language 
            in smart ways.
        
        >>> Attention Mechanism:
            LLMs use attention to capture long-range dependencies between words, enabling them to understand
            context.

        >>> Autoregressive Text Generation:
            LLMs generate text on previously generated tokens, allowing them to produce text in different
            styles and languages.


🧬  large-language-models-3-1: What is special about large language models? Select all that apply.

🧬  A. They use an attention mechanism to capture long-range dependencies between words.
🧬  B. They are pre-trained on vast amounts of data.
    C. They can only generate text in one language.
🧬  D. They use transformer model architecture.
    E. They have to be manually coded for each task.

✔️
Correct! The attention mechanism helps LLMs capture long-range dependencies between words.
Correct! LLMs are pre-trained on vast amounts of data to achieve general-purpose language understanding.
Correct! LLMs are based on transformer model architecture.


**********************************************************************************************************
                                🧬_🌱 The Evolution of Language Models_🧬
**********************************************************************************************************


    The modern capacity of Large Language Models did not arise out of nothing. They have been developed
    for decades, starting with n-gram models.


🧬  N-gram Models

    Early language models used statistical methods that counted how often word sequences appear in text.

    The N in 'N-gram' refers to the number of words that were taken into account. 
    A 2-gram model would only look at sequences of two words, a 3-gram model only at sequences of three words, 
    and so on.

    For example, a 'bigram-model' (2-gram) would look at pairs of words like "New", "York") and estimate how 
    likely "York" is to follow "New".


🧬  large-language-models-4-1: What is a key limitation of N-gram models when it comes to understanding language?

    A. They are unable to process any contextual information
🧬  B. They struggle with long-range context due to limited word windows
    C. They can only process one word at a time
    D. They require too much computational power to be practical


✔️ Correct! N-gram models struggle with long-range context due to limited word windows.

    

🧬  Recurrent Neural Networks

    With the rise of neural networks, models like 'recurrent neural networks' (RNNs) and 'Long Short-Term Memory 
    (LSTMs) impoved the ability to handle longer contexts.

    >>> These models use ''hidden states'' to keep track of information gathered so far. This allows them to
        remember information from earlier in the sequence and use it to make predictions later on.


            >>> hidden state:

                Imagine you're reading a book, sentence by sentence. At any moment, your memory of what you've
                already read helps to understand the current sentence. This memory corresponds to the 'hidden state'.

                Each time your read a new sentence, you update your memory by combining what you just read
                with what you already know.

            Still the models process text sequentially, making it hard to scale to very long inputs.



🧬  Transformers

    The intruduction of the transformer structure was a game-changer.

    Instead of processing words one by one, transformers look at all the words in a sentence simultaneously, 
    determining which words hould "pay attention" to which other words.

    This approach enabled models to learn complex language patterns more efficiently and at larger scales.

    >>> transformers:
        process all words in a sentence simultaneously 

    >>> n-gram models:
        counts how often sequences of n words appear in text

    >>> recurrent neural networks:
        use 'hidden states' to keep track of information gathered so far

        
**********************************************************************************************************
                                            🧬_🤖 Transformers_🧬
**********************************************************************************************************


🧬  transformers architecture:

    https://academy.masterschool.com/ns/books/published/swe/_images/transformers.webp

    
    The transformers architecture is the backbone of almost all modern large language models.

    >>> It relies heavily on the concept of 'attention'

    

🧬  Basic of the Attention Mechanism

    Attention allows the model to focus on specific parts of the input when predicting the next word.

    >>> Suppose you have a sentence:
        
        "The cat sat on the mat, and then it jumped off."

    If you want to understand what "it" refers to, attention mechanisms help the model realize that "it" 
    corresponds to "earlier" in the sentence.

    

🧬  The Innovation of "Self-Attention"

    "Self-attention" means each word in a sentence looks at every other word to decide what is relevant.

    This replaces the sequential processing of older models with a more holistic view of the sentence.

    As a result, transformers can handle long sentences, capture more subtle patterns, and scale up
    to very large datasets.



🧬  Weights

    In practical terms, self-attention is implemented using mathematical operations that produce a set
    of weights.

    These weights tell the model how much to focus on each part of the input.

    Because these operations can be parallelized and are very efficient on modern hardware, training
    huge models on massive text datasets hase become feasable.



🧬  large-language-models-5-1: What is true about transformers and the attention mechanism? Select all that apply.

🧬  A. Transformers rely on the concept of attention.
🧬  B. Self-attention allows each word to look at every other word.
    C. Transformers can only handle short sentences.
🧬  D. Weights are used to determine how much attention to give to each part of the input.
    E. Self-attention is a sequential process.

✔️
Correct! Transformers rely heavily on the concept of attention.
Correct! Self-attention allows each word to look at every other word in the sentence.
Correct! Weights are used to determine how much attention to give to each part of the input.


**********************************************************************************************************
                                🧬_⚙️ How Large Language Models Work_🧬
**********************************************************************************************************



A representation of a neural network. Each circular node represents an artificial neuron and an 
arrow represents a connection from the output of one artificial neuron to the input of another. 
(Source: wikipedia.)

https://academy.masterschool.com/ns/books/published/swe/_images/neural_network.png



🧬  Building and training LLMs is a complex process that involves several steps.


🧬   Building a Large Language Model

    Large Language Models are built using deep learning techniques, which involve neural networks with 
    billions of adjustable settings (parameters).

    These models rely on special structures called 'transformers', which prcess language in smart ways.

    The specific design chosen plays a big role what the model can do and how well it performs tasks
    like translating languages, writing stories, or answering questions.

    

🧬  Training a Large Language Model

    The entire process from collecting data to deploying the model involves several steps.

    >>> Here is a high-level overview of the training process:

    
        >>> choose training data from internet, books, articles, etc.
                
            First, data is collected from various sources like books, articles and websites. 
            This data is then cleaned and processed to remove errors and irrelevant information,
            and to make it easier for the model to learn.

            
        >>> preprocess data: tokenization, embeddings, cleaning...
        >>> use deep-leaning to train model on data

            Next, the model is trained using deep learning techniques. In the fine-tuning stage, 
            the model's parameters are adjusted to minimize errors and improve performance.

            
        >>> evaluate accuracy
        >>> deploy model for use in applications

            Finally, the model's performance is evaluated to ensure it meets the desired criteria.
            If it does, the model is deployed for use in various applications.



🧬  Tokens

    Before a model can process text, the text is broken down into 'tokens'.

    Tokens can be words, subwords, or even characters. Yo can store a sequence of tokens in a python list.

    For instance, if you're given the sentence "I love apples", one way you could tokenize it would be 
    into ["I", "love", "apple", "s"]. the final 's' in 'apples' is a seperate token because it's a 
    plural marker. This, however, is not the only possibility.


    The choice of how to tokenize depends on the model's vocabulary and tokenizer.


🧬  large-language-models-7-1: How many tokens are there in the sentence “The quick brown fox jumps over the lazy dog”?

🧬  A. It depends on the model’s vocabulary and tokenizer.
    B. The sentence contains 9 tokens. One for each word.
    C. 10 tokens. One for each word and one more for the 's' in 'jumps' which indicates the third person.


✔️ Correct! The choice of how to tokenize depends on the model’s vocabulary and tokenizer.


**********************************************************************************************************
                                            🧬_↗ Embeddings_🧬
**********************************************************************************************************


Once the data has been split into tokens, we're ready to try to make sense of these tokens.

The language model converts each token into a numerical representation called a 'vector embedding'. 
A vector embedding is a list of numbers that represents the token's meaning and context.

For example, the vector embedding for "dog" could be [3, 5, 0, 7]. Each number in the list captures
a different aspect of the word's meaning. The first number might represent the word is a noun, the second
stands for an animal, the third that it is close to human habitats, and so on.

This may all seem a bit arbitrary, but once we encode many words this way, we can start to see patterns
and relations between words.



🧬  large-language-models-8-1: Given the encoding of “dog” given in the text, what would the first two numbers 
    in the encoding of “cat” be?

    A. It could be anything, as long as it's a list of numbers.
🧬  B. 3 and 5, because "cat" is also a noun that stands for an animal.
    C. 4 and 6, because "cat" is similar but different from a dog.


✔️ Correct! The first two numbers in the encoding of "cat" would likely be similar to those of "dog" 
    because both are nouns that stand for animals.


**********************************************************************************************************
                                            🧬_🪟 Context Windows_🧬
**********************************************************************************************************


Models don't handle infinitely long text. They have a fixed context window size, meaning they only look
at a certain number of tokens at once.

Older models might have a window of few hundret tokens, while newer ones can handle thousands.

The longer the context window , the more text the model can consider at once, improving its ablility
to maintain coherence and context over long passages.



🧬  large-language-models-9-1: What is a primary benefit of increasing the context window size in language models?

    A. It allows the model to process text faster
🧬  B. It enables the model to consider more text at once, improving coherence
    C. It reduces the model’s memory usage
    D. It allows the model to generate longer responses


✔️ Correct! A larger context window enables the model to consider more text at once, improving coherence.



**********************************************************************************************************
                                🧬_Inference Process: Next-Word Prediction_🧬
**********************************************************************************************************


🧬  When you use an LLM (like typing a prompt into ChatGPT), the model runs in "inference" mode.

    Given the text you've typed so far, it predicts the next token by looking at probabilities
    learned during training.

    It then adds that token to the text and repeats the process to generate the following token, 
    and so forth.

    The process continues until the model reaches a stopping condition, such a maximum length or 
    a special stop token.




🧬  large-language-models-9-2: How does a large language model generate text during the inference process?

    A. The model generates text by randomly selecting words
🧬  B. The model predicts the next token based on probabilities learned during training
    C. The model generates text by looking at the entire input at once
    D. The model generates text by copying and pasting from a database

✔️ Correct! The model predicts the next token based on probabilities learned during training.


**********************************************************************************************************
                                🧬_🛠️ Examples of Large Language Models_🧬
**********************************************************************************************************


Let's look at some examples of large language models that have made siginificant contributions to the field
of natural language processing.


🧬  BERT and its variants

    BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based model that
    learns context from both directions of a text sequence/considers the entire context of a sentence
    at once.

    BERT is often used for tasks like question-anwering, sentiment analysis, and NER. It's a great 
    "encoder" model that excels at understanding text but not generating long texts.

    >>> variants:
        Models like RoBERTa, DistilBERT, and ALBERT are based on BERT and have improved performance 
        or smaller versions of BERT that often perform better or run faster.



🧬  GPT Series (OpenAI)

    GPT (Generative Pre-trained Transformer):
    GPT models are known for their generative capabilities.

    >>> GPT-2 and GPT-3:
        gained widespread attention for producing coherent, human-like text and performing well on a range
        of language tasks with minimal fine-tuning.

    >>> GPT-4 and beyond:
        Newer versions have larger context windows, are more accurate, and can handle complex reasoning 
        tasks.


        
🧬 LLaMA, Claude, and others:

    LLaMA (Language Model for Medical Applications):
    A model trained on medical text that can help with tasks like medical diagnosis and information retrieval.

    Claude:
    A model trained on code that can help with tasks like code completion and bug detection.

    These models show the versatility of LLMs and their potential to revolutionize various industries.


🧬  These models each have unique features, but they are all based on transformer architecture and 
    pre-training/fine-tuning paradigm. Their capabilities are continually improving, expanding the range
    of tasks they can handle and the qualitiy of their responses.




**********************************************************************************************************
                                        🧬_✅ Chapter Assessment_🧬
**********************************************************************************************************


As you advance in your AI learning journey, knowing how large language models work and how to use them 
will be invaluable.


    🧬  In-Depth Reasoning:
        LLMs help you to understand how AI can grasp context, meaning and nuance in human language.

        
    🧬  Practical Skills:
        With straightforward APIs and libraries, you can start integrating LLMs into software 
        projects to build chatbots, summarizers, code assistants and much more.


    🧬  Future Opportunities:
        LLMs will continue evolving, and understanding their foundations prepares you to adapt and take
        advantage of emerging opportunities in AI.


        

🧬  Let's go through a quick check to see how well you've grasped the concepts covered in this chapter:




🧬  large-language-models-11-1: Why do language models learn probability distributions of sequences of words?

    A. It is easiest to focus on words that are close to each other.
    B. It is easiest to focus on individual words.
    C. Language is inherently non-sequential.
🧬  D. Language is inherently sequential.


✔️ Correct! Language is inherently sequential, and the order of words matters.






🧬  large-language-models-11-2: What are common uses of large language models? Select all that apply.

🧬  A. Translating between natural languages.
    B. Compiling code from different programming languages.
🧬  C. Generating human-like text.
    D. Analyzing images and videos.


✔️
Correct! Large language models can be used for translating between natural languages.
Correct! Large language models can generate human-like text.





🧬  large-language-models-11-3: What is the role of tokens in training large language models?

    A. Tokens represent individual data points in audio recordings.
🧬  B. Tokens are small pieces of text, like words or characters, that the model processes.
    C. Tokens are special markers used to identify errors in a model.
    D. Tokens are the mathematical operations used to train the model


✔️ Correct! Tokens are small pieces of text, like words or characters, that the model processes.




🧬  large-language-models-11-4: What are embeddings in the context of large language models?

    A. A way to compress the model's memory for efficient storage.
🧬  B. A technique for converting words or tokens into numerical vectors that capture their meaning and relationships.
    C. A set of rules the model uses to generate grammatically correct sentences.
    D. A process for dividing text into smaller, easier-to-process segments.


✔️ Correct! Embeddings convert words or tokens into numerical vectors that capture their meaning and relationships.





🧬  large-language-models-11-5: What is fine-tuning in the context of large language models?

    A. Testing the model with new datasets to measure accuracy.
    B. Adjusting the model's predictions based on specific user feedback.
    C. Fixing bugs in the model's code to make it run faster.
🧬  D. Training the model further on specialized data to improve performance in a specific area.


✔️ Correct! Fine-tuning involves training the model further on specialized data to improve performance in a specific area.




**********************************************************************************************************
                                        🧬_Generative AI APIs in Practice!_🧬
**********************************************************************************************************


🧬  Modern Large Language models (LLMs) are remarkably flexible and can be accessed through various APIs.
    So far we've been mostly looking into the theoretical aspects of LLMs. in this practical lesson, we will
    walk through four popular GenAi API examples to illustrate different concepts:

    1. Simple generation

    2. Using system and user prompts

    3. Adjusting temperature and max tokens

    4. Maintaining conversation history


🧬   We will also look at how we can implement our own language models locally with 'Hugging Face transformers'.
     Each exercise includes a short code snippet, a brief explaination of the key concepts, and a small exercise 
     that invites you to explore the code further.


🧬  API Keys and Credits
    Rules for using these services can change over time. Some might give you a free trial, others may offer 
    starter credits, and a few might require payment from the get-go. Regardless, you’ll need an API key 
    to unlock their features. We encourage you to explore them all—you’ll gain a better sense of 
    the incredible possibilities they offer!

    
**********************************************************************************************************
🧬__🧬
**********************************************************************************************************


🧬  Using an API or Open-Source Model

    >>> OpenAI's GPT-3 API
    >>> Hugging Face's Transformers Library
    >>> Google's BERT API
    >>> Microsoft's LUIS API
    >>> IBM's Watson API
    >>> Facebook's PyTorch API
    >>> Amazon's Lex API
    >>> Salesforce's Einstein API
    >>> And many more!

    All the APIs we're exploring today come with dedicated websites and documentation, and they're definitely
    worth a look! One reason is that, as we mentioned, the implementation of models can revolve over time. 
    Another is that diving into docs can reveal a range of possibilities, like different models, settings,
    and usage instructions, that you might find inspiring.




🧬  OpenAI API
    
        >>> OpenAI's GPT-3 API is one of the most popular and powerful language models available today.
        >>> It can generate human-like text, answer questions, and even write code.
        >>> The API is accessible via a RESTful interface, making it easy to integrate into web applications
            and other services.

    Allows you to send text prompts to an LLM like GPT-4o and receive generated completions. It's a great starting
    point for beginners because you don't need to train or manage the model - just provide a prompt and read
    the output.

    OpenAI offers great documentation for the API. Have a look at these:

    >>> https://platform.openai.com/docs/guides/text-generation
    >>> https://platform.openai.com/docs/api-reference/completions/create
    >>> https://platform.openai.com/docs/api-reference/chat

    


🧬  Anthropic Claude API


    >>> Enables developers to interact with advanced conversational AI models, such as Claude 3.5 Sonnet, 
        by sending text prompts and receiving generated responses.

    >>> Anthropic Claude is a language model trained on code that can help with tasks like code completion,
        bug detection, and code summarization.
    >>> The API is accessible via a RESTful interface, making it easy to integrate into web applications
        and other services.
    
    Allows you to send code snippets to the model and receive completions, suggestions, or bug fixes. It's a
    great tool for developers looking to improve their coding workflow.

    Anthropic offers great documentation for the API. Have a look at these:

    >>> https://docs.anthropic.com/claude/api-reference
    >>> https://docs.anthropic.com/claude/quickstart?lang=python
    >>> https://docs.anthropic.com/en/docs/initial-setup
    >>> https://docs.anthropic.com/en/api/getting-started



🧬  Google Gemini API

    >>> Google's Gemini API is a powerful language model that can generate human-like text, answer questions,
        and provide recommendations.
    >>> The API is accessible via a RESTful interface, making it easy to integrate into web applications
        and other services.
    
    >>> Google's Gemini API allows developers to access the latest generative AI models, such as Gemini 1.5
        Flash and Gemini 1.5 Pro, by sending text prompts and receiving generated responses.

    >>> https://ai.google.dev/gemini-api/docs/quickstart?lang=python
    >>> https://ai.google.dev/gemini-api/docs/text-generation?lang=python
    >>> https://ai.google.dev/gemini-api/docs/api-reference/completions/create?lang=python
    >>> https://ai.google.dev/gemini-api/docs/api-reference/chat?lang=python



🧬  Groq API

    While not an LLM provider per se, Groq offers a hardware and software stack optimized for running large
    models efficiently.

    Documentation on integrating LLMs with Groq hardware solutions can be found on their website, 
    guiding developers on deploying large scale models in production environments.

    >>> https://groq.com/docs/groq-ai-platform
    >>> https://groq.com/docs/groq-ai-platform/quickstart
    >>> https://groq.com/docs/groq-ai-platform/api-reference
    >>> https://groq.com/docs/groq-ai-platform/chat

    >>> https://console.groq.com/docs/quickstart
    >>> https://console.groq.com/docs/text-chat



🧬  Hugging Face

    Provides open-source models you can download and run locally. tools like transformers library in 
    Python let you load a pre-trained model and generate text easily.

    >>> https://huggingface.co/transformers/
    >>> https://huggingface.co/transformers/quicktour.html
    >>> https://huggingface.co/transformers/usage.html
    >>> https://huggingface.co/transformers/model_doc/gpt.html
    >>> https://huggingface.co/transformers/model_doc/claude.html
    >>> https://huggingface.co/transformers/model_doc/gemini.html
    >>> https://huggingface.co/transformers/model_doc/groq.html


    >>> https://huggingface.co/
    >>> https://huggingface.co/docs/transformers/en/main_classes/pipelines.html


**********************************************************************************************************
                                🧬_Simple Generation with 'Google Gemini'_🧬
**********************************************************************************************************


🧬 Key Concept:

    The first LLM API we'll look at today is Google's Gemini API. We will use it to demonstrate a basic
    text generation request using a single user prompt.


🧬  Quickstart:

    install the google-generativeai package using the following pip command:

    >>> pip install google-generativeai


🧬  Explanation:

    1. Confuguration: You set the API key using 'genai.configure(...)'. 
        To get an API key go to https://ai.google.dev/gemini-api/docs/api-key.

    2. Select a Model: "gemini-2.0-flash-exp" is provided as the generative model.

    3. User Prompt: A single question ("What is GenAI?") is passed to the model.

    4. Response: The model returns generated text based on its trained knowledge.


    
🧬  Exercise:

    1. Change the prompt to something else, like “Explain the basics of data structures in Python.”

    2. Try using a different model, like "gemini-2.0-pro-exp" or "gemini-2.0-pro-adv".

    
**********************************************************************************************************
                                    🧬_System and User Prompts with Groq_🧬
**********************************************************************************************************


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