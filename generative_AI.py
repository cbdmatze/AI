'''
**********************************************************************************************************
                                            🧬_🤖 What is AI?_🧬
**********************************************************************************************************


🧬  Welcome to the first lesson on Generative AI!

Generative AI is about enabling machines to create new data - images, text, audio - based on patterns 
learned from existing data.

It is a particularly exciting field of AI that has seen rapid advancements in recent years - you 
probably heard about GPT-3, DALL-E, and other generative models.


🧬  So hop on board and explore the world of Generative AI. By the end of this lesson, 
    you will be able to explain what Generative AI is, identify its key technologies, 
    and understand why it is a driving force in modern technology and software engineering.

    
**********************************************************************************************************
                                    🧬_📖 Definition and Overview_🧬
**********************************************************************************************************


🧬  Nowadays when people speak of AI they often mean machine learning, and more specifically, 
    deep learning. But let’s be precise and distinguish between these terms!

    
🧬  Artificial Intelligence (AI):

    is a branch of computer science that aims to create machines capable of performing tasks that normally
    require human intelligence, such as visual perception, speech recognition, decision-making, and language
    translation.

    
    AI is more thatn just machine learning!

    A program can be considered AI if it performs a task that would typically require human intelligence. 
    This doesn't have to involve machine learning. For example, a chess program that uses a set of rules
    to determine the best moves is an AI program, but doesn't use machine learning. 


🧬  Machine Learning (ML):

    is a subset of AI that focuses on developing algorithms and statistical models that enable computers
    to learn patterns from data and make predictions or decisions without being explicitly programmed.


🧬  Deep Learning (DL):

    is a specialized subset of ML that uses artificial neural networks with multiple layers to learn 
    representations of data. You will learn about these concepts soon.


🧬  Generative AI (GenAI):

    is a specific application of deep learning that involves training models to generate new data samples,
    based on patterns learned from existing data. 


    Here you can see how these fields relate to each other:

    https://academy.masterschool.com/ns/books/published/swe/_images/circles.png


    
computers mimic or surpass human cognitive functions -> AI

algorithms that learn patterns from data -> ML

multi-layered neural networks to learn representations of data -> DL

training models to generate new data samples based on patterns learned from existing data -> GenAI


**********************************************************************************************************
                                    🧬_🤔 What is Generative AI?_🧬
**********************************************************************************************************


🧬  Generative AI refers to systems that can create new data similar to what they’ve been trained on. 
    Instead of just recognizing patterns, generative models can produce entirely original text, 
    images, music, or other forms of content.


    Note

    Traditional AI models are discriminative, meaning they learn to classify or predict based on existing 
    data. For example, a model can be trained on data of cat pictures, and when shown a new image, 
    it can predict whether it contains a cat.

    Generative AI, on the other hand, can generate new cat pictures that look like they could be real! 
    This goes beyond classification or recognition tasks; generative AI has the creative ability 
    to “imagine” new content.


🧬  In recent years, generative models have become increasingly sophisticated, enabling them to 
    generate high-quality, realistic content across various domains. A generative text model, 
    when prompted with a few words, can generate a coherent paragraph that reads as though a human 
    wrote it.



**********************************************************************************************************
                        🧬_✍️ Text Generation (e.g., ChatGPT, Claude.ai):_🧬
**********************************************************************************************************


🧬  Text generation models are trained to understand and generate human-like language. 
    The ones that are currently successful, like GPT, are Large Language Models (LLMs) which can write 
    essays, answer questions, summarize documents, and engage in conversation.




🧬  It’s called a large language model because it is trained on large amounts of text data, 
    such as books and online articles. The model uses a neural network to process sentences 
    in chunks (called tokens) and learns patterns, meanings, and relationships between words.




🧬  During training, the model predicts what word is likely to come next in a sentence and adjusts 
    itself to improve accuracy. Once trained, it can perform tasks like answering questions or writing 
    summaries by applying its understanding of language and context.


**********************************************************************************************************
                    🧬_🖼️ Image Generation (e.g., DALL·E, MidJourney):¶_🧬
**********************************************************************************************************

    
🧬  Image generation models can create original images based on textual prompts. For example, 
    you could say “Create an image of a robot reading a book in a cozy library,” and a generative 
    model might produce a new image that matches this description. Such tools inspire artists, 
    help in rapid prototyping of product designs, and fuel creativity in marketing and advertising.




🧬  The way these models work is similar to text models, but instead of predicting words, 
    they predict pixels in an image. They learn to generate images by analyzing large datasets 
    of images and understanding the relationships between different parts of an image.




🧬  Whereas the relations in a text are sequential, relations in an image are spatial, so these models 
    are trained to recognise spatial patterns.


**********************************************************************************************************
                                🧬_💻 Code Generation (e.g., GitHub Copilot):_🧬
**********************************************************************************************************


🧬  These models assist developers by suggesting code snippets, providing function implementations, 
    and sometimes even generating complete modules. They can speed up software development, 
    help in learning new programming languages, and improve code quality.




🧬  They are very similar to the models used for natural language; after all, code is also a kind of 
    language. But they are trained especially to recognise the hierarchical structure of code.



**********************************************************************************************************
                                    🧬_🎵 Music, Art, and Game Design:_🧬
**********************************************************************************************************


🧬  Generative AI can compose music, design video game levels, or create digital art.
    It enables rapid experimentation, provides endless variations of creative content, and allows 
    smaller teams to produce high-quality creative assets that previously required large 
    specialized teams.

    

    🧬_Check your understanding_🧬

    intro-genai-nlp-2-1: How does generative AI differ from traditional AI?

    A. GenAI can work with images and audio.
    B. GenAI can only recognize patterns in data.
    C. GenAI can only classify or predict based on existing data.
🧬  D. GenAI can generate new data similar to what it has been trained on.


**********************************************************************************************************
                                        🧬_🔗 Neural Networks_🧬
**********************************************************************************************************


🧬  Neural networks are computational models inspired by the structure of the human brain,
    consisting of interconnected layers of artificial “neurons” that process information.




🧬  Each neuron takes input, applies a transformation, and passes the output to the next layer. 
    The result depends on the connections between neurons and the weights assigned to these connections.




🧬  By adjusting the connections (weights) between these neurons, a network can learn complex patterns 
    from data. Neural networks form the backbone of modern AI applications.



**********************************************************************************************************
                                        🧬_📚 Deep Learning_🧬
**********************************************************************************************************



🧬  Deep learning refers to neural networks with many layers (“deep” architectures). T
    hese additional layers allow the model to learn more nuanced, high-level representations of data. 
    For example, in image generation, early layers might learn to detect edges and colors, 
    while deeper layers learn to recognize objects and composition.




🧬  If you think about it, that is how our brains work, too: we process simple features first (like shapes and colors) and then combine them to recognize complex objects (like faces or animals).


🧬_intro-genai-nlp-3-1: Which statements are true about neural networks, deep learning, 
    and transformers? Select all that apply._🧬


🧬  A. Neural networks are inspired by the structure of the human brain.
🧬  B. Deep learning refers to neural networks with many layers.
    C. Transformers process input data sequentially.
🧬  D. Transformers use an attention mechanism to focus on different parts of the input.
    E. Neural networks can only learn simple patterns from data.


Correct! Neural networks are inspired by the structure of the human brain.
Correct! Deep learning refers to neural networks with many layers.
Correct! Transformers use an attention mechanism to focus on different parts of the input.


**********************************************************************************************************
                                        🧬_⚡ Transformers_🧬
**********************************************************************************************************



🧬  Transformers are a type of deep learning architecture that has revolutionized natural language 
    processing and other areas. Unlike earlier neural network architectures, transformers handle 
    input data (like words in a sentence) all at once rather than sequentially.




🧬  They use a mechanism called attention that allows the model to focus on different parts of 
    the input dynamically. This has enabled training on massive datasets and paved the way for large 
    language models (LLMs) like GPT.


**********************************************************************************************************
                                    🧬_🌍 Current and Future Impact_🧬
**********************************************************************************************************



🧬  You want to learn about generative AI? This is a great time to do so!


🧬  Generative AI is already reshaping industries and professional workflows. By starting your journey 
    now, you position yourself at the forefront of a field that will only grow in importance.


**********************************************************************************************************
                                    🧬_Why Learn Generative AI?_🧬
**********************************************************************************************************


🧬  Marketers use AI to generate personalized campaign content


🧬  Generative AI has revolutionized marketing by enabling highly personalized content creation.
    Marketers can use AI to craft tailored emails, advertisements, and social media posts that 
    resonate with specific audiences.


🧬  By analyzing consumer behavior and preferences, these tools can generate compelling copy 
    and visuals designed to maximize engagement and conversion rates. This not only saves time 
    but also allows businesses to reach diverse segments of their audience with messages 
    that feel custom-made, fostering stronger customer relationships.


🧬  Developers save time with code generation


🧬  For developers, generative AI tools like GitHub Copilot or ChatGPT have become invaluable, 
    acting as virtual coding assistants. These tools can suggest, write, and debug code snippets, 
    drastically reducing the time spent on repetitive tasks.


🧬  By automating mundane aspects of programming, developers can focus on solving complex problems 
    and creating innovative solutions.


🧬  Researchers use it for rapid prototyping and hypothesis testing




🧬  In research, generative AI accelerates innovation by facilitating rapid prototyping and exploration 
    of new ideas. Scientists and academics can use AI to generate data models, simulate scenarios, 
    or draft initial hypotheses.


🧬  For instance, in drug discovery, AI models can predict molecular interactions, reducing the time 
    needed for laboratory experiments.


🧬  Similarly, in academic research, generative AI aids in drafting papers and summarizing literature.


🧬  By streamlining the early stages of research, these tools allow researchers to focus on analysis 
    and experimentation, driving faster progress in their fields.


**********************************************************************************************************
                            🧬_💻 Role in Software Engineering and Other Industries_🧬
**********************************************************************************************************


🧬  In software engineering, understanding generative models can help you integrate intelligent 
    features into applications, automate tasks, and create more intuitive user interfaces.




🧬  Outside of software, knowledge of generative AI is valuable in healthcare (for synthesizing
    medical data), finance (for predictive modeling and analysis), manufacturing (for designing 
    new products), and countless other domains.


**********************************************************************************************************
                            🧬_🤖 What is Natural Language Processing (NLP)?_🧬
**********************************************************************************************************


🧬  Let’s talk about natural language processing! This is one of the more exciting areas in the field 
    of artificial intelligence, which has seen lots of progress in recent years.


🧬  NLP is a field of AI that focuses on enabling computers to understand, interpret, and generate 
    human language. This includes both written text and spoken language. NLP systems aim to bridge 
    the gap between human communication and machine understanding, allowing software to interact 
    with users in a more natural and intuitive manner.


**********************************************************************************************************
                            🧬_🔍 Difference Between NLP, NLU, and NLG_🧬
**********************************************************************************************************


🧬  NLP (Natural Language Processing): The broad field encompassing the entire pipeline of 
    understanding and generating language.


🧬  NLU (Natural Language Understanding): A subset of NLP focused on interpreting the meaning 
    and intent behind human language. This includes tasks like part-of-speech tagging, 
    named entity recognition, and semantic understanding. Essentially, NLU tries to 
    “read” and “comprehend” language.


🧬  NLG (Natural Language Generation): Another subset of NLP dedicated to producing coherent text 
    in a human language. NLG takes structured information (like data from a spreadsheet) 
    or internal model representations and turns it into understandable, fluent sentences. 
    This is what powers many chatbots, summary generators, and language models.



    
🧬  A. NLP focuses on enabling computers to understand, interpret, and generate human language.
🧬  B. NLP combines linguistics, computer science, and machine learning.
    C. NLP is a subset of NLU.
    D. NLP uses transformer architectures

✔️
Yes, NLP is about understanding, interpreting, and generating human language.
Correct, NLP combines these fields to achieve its goals.
Correct, transformer architectures are commonly used in modern NLP systems.


**********************************************************************************************************
                                    🧬_🌐 Applications of NLP_🧬
**********************************************************************************************************


🧬  What are some of the key applications of Natural Language Processing (NLP)? Let’s explore the 
    exciting ways NLP is used in the real world.


**********************************************************************************************************
                                    🧬😊 Sentiment Analysis__🧬
**********************************************************************************************************


🧬  Sentiment analysis involves determining the emotional tone behind a piece of text. For example, 
    when analyzing product reviews or social media posts, companies use sentiment analysis to gauge 
    customer feelings—positive, negative, or neutral—towards their products or services.


**********************************************************************************************************
                                        🧬_🌍 Machine Translation_🧬
**********************************************************************************************************


🧬  Machine translation tools convert text from one language to another, bridging language gaps and 
    making information accessible globally. Models like Google Translate or DeepL have become 
    increasingly accurate, enabling people around the world to communicate more easily.


**********************************************************************************************************
                                🧬_🗣️ Speech-to-Text and Text-to-Speech_🧬
**********************************************************************************************************


🧬  NLP techniques are used to transcribe spoken words into text and to generate natural-sounding 
    speech from written text. These technologies power digital assistants like Siri and Alexa, 
    enable dictation systems, and assist those with visual or hearing impairments.


**********************************************************************************************************
                                🧬_🤖 Chatbots and Virtual Assistants_🧬
**********************************************************************************************************


🧬  Many websites and applications employ chatbots that can answer frequently asked questions, 
    guide customers through support processes, or even schedule appointments. Virtual assistants 
    on smartphones and smart speakers can set reminders, send messages, and control smart home 
    devices using voice commands.


**********************************************************************************************************
🧬__🧬
**********************************************************************************************************


🧬  Named Entity Recognition: 

    Identifying specific entities in text, like names of people, places, organizations, or dates.

🧬  Summarization: 

    Generating concise summaries of long documents to help users quickly understand the main points.

🧬  Information Retrieval and Q&A Systems: 

    Retrieving relevant documents or information from databases based on user queries and sometimes 
    directly answering questions using AI models.


**********************************************************************************************************
                        🧬_📚 Why are These Topics Important for You as a Beginner?_🧬
**********************************************************************************************************


🧬  Understanding Generative AI and NLP is essential because these technologies represent some of the most visible and impactful areas of AI today. They provide the foundation for creating intelligent, user-friendly interfaces and open up a multitude of career paths.


🧬  As you progress, you’ll learn how to build models, fine-tune pre-trained systems, and creatively apply these technologies to solve real-world problems. By starting with the fundamentals, you are laying a strong groundwork that will serve you as you move into more advanced courses and specializations.



📌 In Summary
✨ Generative AI empowers machines to create original content—text, images, code, music—opening doors to endless possibilities in creativity, productivity, and innovation.

🧠 NLP enables computers to understand and generate human language, making technology more accessible, intuitive, and human-centered.

🔧 As you move forward in this field, you will gain the skills and knowledge to build on these foundations, adapt to new developments, and contribute to the evolving landscape of AI-driven solutions.


**********************************************************************************************************
                                        🧬_📝 Chapter Assessment_🧬
**********************************************************************************************************


🧬  intro-genai-nlp-8-1: 1. Which of the following best describes Artificial Intelligence (AI)?


    A. A narrow field that focuses only on text analysis.
🧬  B. A broad field aiming to create machines that mimic human intelligence.
    C. A technology used solely for image recognition tasks.
    D. A hardware solution for faster data processing.

    
    ✔️ B is correct because AI is indeed a broad field concerned with building systems that exhibit human-like intelligence, including reasoning, learning, and problem-solving.



🧬  intro-genai-nlp-8-2: 2. What differentiates Machine Learning (ML) from Deep Learning (DL)?


    A. ML cannot learn from data, while DL can.
    B. ML is a subset of AI, while DL is not related to AI.
🧬  C. DL uses many-layered neural networks to learn complex patterns; ML includes a broad range of techniques.
    D. ML models always outperform DL models.

    ✔️ C is correct because Deep Learning is a subset of Machine Learning that specifically involves neural networks with many layers.



🧬  intro-genai-nlp-8-3: 3. What is Generative AI primarily known for?


    A. Classifying existing images into different categories.
    B. Summarizing existing text without adding new information.
🧬  C. Creating new, original content such as text, images, or music.
    D. Only analyzing big data without producing output.

    ✔️ C is correct; Generative AI produces new, original content similar to what it was trained on.



🧬  intro-genai-nlp-8-4: 4. Which of the following is an application of Generative AI?


    A. Sentiment analysis of text reviews.
    B. Translating text from English to French.
🧬  C. Generating new images from text prompts (e.g., DALL·E).
    D. Recognizing if an image contains a dog or a cat.

    ✔️ C is correct because image generation from text prompts is a classic example of Generative AI.

    


🧬  intro-genai-nlp-8-5: 5. What role do Transformers play in modern AI?


    A. They are hardware devices used to store big data.
🧬  B. They are a type of neural network architecture that uses attention mechanisms to process input.
    C. They are simple rule-based systems for grammar checking.
    D. They are only used for image processing tasks.

    ✔️ B is correct because Transformers are a neural network architecture that revolutionized NLP and other fields with their attention-based mechanism.


🧬  intro-genai-nlp-8-6: 6. Why is Generative AI important to study?


    A. Because it has no practical applications and is purely theoretical.
    B. Because it has no relevance to future technology trends.
🧬  C. Because it helps generate new content, impacting industries from design to software engineering.
    D. Because it replaces all traditional programming languages.

    ✔️ C is correct; Generative AI is already shaping multiple industries, and its influence is growing.



🧬  intro-genai-nlp-8-7: 7. What is Natural Language Processing (NLP)?


🧬  A. A field that deals with analyzing and generating human language.
    B. A hardware technique for processing signals.
    C. A method for designing mechanical systems.
    D. A type of database technology.

    ✔️ A is correct; NLP is the field of AI focused on understanding and generating human language in both written and spoken forms.




🧬  intro-genai-nlp-8-8: 8. How does Natural Language Understanding (NLU) differ from Natural Language Generation (NLG)?


🧬  A. NLU is about understanding the meaning in text, while NLG is about producing new text.
    B. NLU focuses on producing speech sounds, while NLG focuses on storing data.
    C. NLU translates text between languages, while NLG only detects sentiment.
    D. NLU and NLG are the same thing.

    ✔️ A is correct; NLU deals with interpreting language, while NLG deals with creating it.



🧬  intro-genai-nlp-8-9: 9. Which of the following is an application of NLP?


    A. Generating entirely new images without reference images.
🧬  B. Converting spoken words into written text.
    C. Predicting stock prices based on market data alone.
    D. Controlling robotic arms in a factory.

    ✔️ B is correct; speech-to-text is an NLP task because it deals with understanding and transcribing human language.


🧬  intro-genai-nlp-8-10: 10. Which of the following best explains why Transformers improved NLP performance significantly?


    A. They rely only on keyword searches rather than neural networks.
🧬  B. They process language inputs all at once and use attention to focus on important parts of the text.
    C. They remove all complexity by using simple rule-based grammar checks.
    D. They only work on very small datasets.

    ✔️ B is correct; Transformers handle the entire input sequence simultaneously and use attention mechanisms to identify which parts of the input are most relevant.


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

















'''