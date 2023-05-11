import pandas as pd
from numpy import dot
from numpy.linalg import norm
import numpy as np
import gensim
from scipy.spatial import distance
from tqdm import tqdm
import nltk
import string
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords

def load_model():
    return gensim.models.KeyedVectors.load_word2vec_format('./wiki.en.vec')

topic_keywords = {
    'ComputerScience': ['algorithms', 'data', 'programming', 'networks', 'databases', 'security', 'software', 'development', 'efficiency','backend', 'frontend', 'web', 'robotics', 'computation', 'computational', 'code', 'run', 'debug', 'input',  'output', 'technology', 'overflow', 'binary', 'bit', 'bits', 'byte', 'bytes', 'computer', 'computers',  'computing', 'program', 'programs', 'programming', 'programmer', 'programmers', 'programmed', 'automation', 'devops', 'cloud', 'architecture', 'servers', 'testing', 'debugging', 'scaling', 'data structures', 'analysis', 'analytics', 'design patterns', 'machine learning', 'deep learning', 'neural networks', 'data science',  'database management', 'database design', 'data mining', 'data modeling', 'data warehousing', 'data visualization',  'data analysis', 'data-driven', 'information', 'information systems', 'artificial intelligence', 'computer vision', 'natural language processing', 'cybersecurity', 'cryptography', 'encryption', 'decryption', 'hacking', 'penetration testing', 'firewalls', 'authentication', 'authorization', 'access control', 'virtualization', 'cloud computing', 'cloud storage', 'distributed computing', 'parallel computing', 'high-performance computing', 'scalability', 'optimization', 'operating systems', 'system administration', 'mobile development', 'web development', 'responsive design',  'user interface', 'user experience', 'agile', 'project management', 'version control', 'continuous integration',  'continuous delivery', 'source code', 'open source', 'intellectual property', 'licensing', 'patents', 'copyright', 'data privacy', 'online privacy', 'e-commerce', 'web services', 'internet of things', 'big data', 'blockchain',  'virtual reality', 'augmented reality', 'quantum computing', '3D printing', 'nanotechnology', 'game development', 'graphics programming', 'rendering', 'compilers', 'interoperability', 'API', 'JSON', 'XML', 'HTTP', 'HTTPS',  'TCP/IP', 'FTP', 'SMTP', 'REST', 'SOAP', 'microservices', 'serverless', 'containerization', 'kubernetes', 'docker', 'ansible', 'terraform', 'jenkins', 'puppet', 'chef', 'salt', 'ansible', 'monitoring', 'logging',  'troubleshooting', 'incident response', 'performance tuning', 'load balancing', 'fault tolerance',  'reliability engineering', 'chaos engineering', 'computer graphics', 'computer vision', 'user research',  'virtual assistants', 'machine translation', 'audio processing', 'speech recognition', 'natural language understanding', 'predictive analytics', 'data preprocessing', 'data cleaning', 'data integration', 'data engineering',  'data governance', 'data quality', 'data enrichment', 'data exploration', 'data validation', 'data profiling', 'data storage', 'data migration', 'data synchronization', 'data access', 'data federation', 'data catalog',  'data lineage', 'data lineage', 'data lineage', 'data transformation', 'data curation', 'data lineage',  'data masking', 'data anonymization', 'data ethics'],
    'Chemistry': ['atoms', 'molecules', 'elements', 'reactions', 'thermodynamics', 'kinetics', 'spectroscopy', 'quantum', 'organic', 'inorganic', 'oxygen', 'co2', 'hydrogen', 'electron', 'electronic', 'electrons', 'proton', 'protons', 'napthol', 'state', 'experiment', 'measurements', 'measurement', 'element', 'acid', 'base', 'pH', 'redox', 'stoichiometry', 'enthalpy', 'entropy', 'gibbs', 'solubility', 'solution', 'equilibrium', 'rate', 'rate law', 'catalysis', 'transition state', 'molecular orbitals', 'periodic table', 'valence', 'bonding', 'hybridization', 'isomers', 'chirality', 'alkanes', 'alkenes', 'alkynes', 'aromatics', 'amines', 'alcohols', 'carbonyls', 'carboxylic acids', 'esters', 'amines', 'amides', 'polymers', 'biomolecules', 'proteins', 'nucleic acids', 'lipids', 'carbohydrates', 'chromatography', 'mass spectrometry', 'infrared spectroscopy', 'UV-Vis spectroscopy', 'NMR spectroscopy', 'X-ray crystallography', 'gas laws', 'ideal gas', 'real gas', 'colligative properties', 'phase diagrams'],
    'Physics': ['waves', 'particle', 'thermal energy', 'fluids', 'electricity', 'nuclear', 'dynamics', 'vibration', 'power', 'general relativity', 'gravitational potential energy', 'potential energy', 'quantum mechanics', 'circuits', 'electrostatics', 'magnetism', 'energy', 'astrophysics', 'particle physics', 'optics', 'temperature', 'kinematics', 'wave', 'mechanics', 'entropy', 'radioactivity', 'physical optics', 'quantum theory', 'kinetic energy', 'subatomic', 'astronomy', 'nuclear physics', 'conservation of energy', 'thermodynamics', 'relativity', 'oscillation', 'solid', 'cosmology', 'work', 'heat', 'electromagnetism', 'thermostat', 'quantum', 'geometrical optics', 'special relativity'],
    'Math&Statistics': ['math', 'mathematics', 'calculus', 'differentiation', 'integration', 'derivatives', 'limits', 'functions', 'graphing', 'equations', 'algebra', 'linear', 'quadratic', 'polynomial', 'exponential', 'logarithmic', 'trigonometric', 'complex', 'vector', 'matrix', 'probability', 'probabilistic', 'statistics', 'statistical', 'data', 'analysis', 'sampling', 'hypothesis', 'testing', 'inference', 'regression', 'correlation', 'ANOVA', 'random', 'variable', 'distribution', 'normal', 'binomial', 'poisson', 'chi-squared', 't-distribution', 'f-distribution', 'confidence', 'interval', 'estimation', 'discrete', 'discretization', 'combinatorics', 'permutation', 'combination', 'graph', 'graphing', 'network', 'theory', 'graph', 'geometry', 'Euclidean', 'non-Euclidean', 'topology', 'fractal', 'dimension', 'metric'],
    'Pharma': ['drugs', 'medicines', 'pharmaceuticals', 'pharmacy', 'pharmacology', 'pharmaceutics', 'pharmacokinetics', 'pharmacodynamics', 'clinical', 'preclinical', 'toxicology', 'pharmacogenetics', 'pharmacogenomics', 'pharmacovigilance', 'pharmacoepidemiology', 'pharmacoeconomics', 'drug interactions', 'drug delivery', 'drug development', 'drug discovery', 'therapeutics', 'therapeutic agents', 'biopharmaceuticals', 'biologics', 'biosimilars', 'generic drugs', 'over-the-counter drugs', 'prescription drugs', 'active pharmaceutical ingredients', 'excipients', 'formulations', 'dosage forms', 'clinical trials', 'drug safety', 'pharmaceutical regulation', 'pharmaceutical marketing', 'pharmaceutical sales', 'pharmacy benefit management'],
    'Biology': ['biology', 'evolution', 'genetics', 'cell', 'ecology', 'physiology', 'neuroscience', 'immunology', 'microbiology', 'biotechnology', 'biochemistry', 'heart', 'lung', 'brain', 'body', 'bone', 'bones', 'muscle', 'muscles', 'blood', 'tissue', 'tissues', 'organ', 'organs', 'organism', 'organisms', 'cellular', 'chromosome', 'gene', 'DNA', 'RNA', 'nucleus', 'mitosis', 'meiosis', 'prokaryote', 'eukaryote', 'adaptation', 'natural selection', 'mutation', 'inheritance', 'variation', 'cloning', 'virus', 'bacteria', 'fungi', 'parasite', 'immunity', 'antibody', 'vaccine', 'antigen', 'pathogen', 'disease', 'infection', 'epidemic', 'endocrine', 'hormone', 'neuron', 'synapse', 'reflex', 'afferent', 'efferent', 'peripheral', 'central', 'cerebellum', 'cerebral cortex', 'neurotransmitter', 'dendrite', 'axon', 'action potential', 'membrane potential', 'synaptic transmission', 'receptor', 'ligand', 'signal transduction', 'endocytosis', 'exocytosis', 'vesicle', 'cytoskeleton', 'flagellum', 'cilia', 'organelle', 'membrane', 'osmosis', 'diffusion', 'active transport', 'enzyme', 'substrate', 'metabolism', 'glycolysis', 'citric acid cycle', 'electron transport chain', 'photosynthesis', 'respiration', 'fermentation', 'amino acid', 'protein', 'carbohydrate', 'lipid', 'nucleotide', 'enzyme', 'hormone', 'neurotransmitter', 'receptor', 'apoptosis', 'cancer', 'tumor', 'stem cell', 'regeneration', 'development', 'differentiation', 'gamete', 'fertilization', 'zygote', 'embryo', 'blastula', 'gastrula', 'morula', 'organogenesis', 'homeostasis', 'feedback', 'positive feedback', 'negative feedback', 'metabolism', 'nutrition', 'digestion', 'absorption', 'excretion', 'circulatory', 'lymphatic', 'respiratory', 'excretory', 'immune', 'nervous', 'endocrine', 'muscular', 'skeletal', 'integumentary', 'reproductive', 'vertebrate', 'invertebrate'],
    'Psychology': ['cognition', 'cognitive', 'perception', 'perceive', 'learning', 'learn', 'memory', 'remember', 'development', 'develop', 'personality', 'personality traits', 'traits', 'psychology', 'psychological', 'social', 'sociology', 'sociological', 'counseling', 'counsel', 'neuropsychology', 'neuroscience', 'neurological', 'behavioral', 'behavior', 'behaviors', 'behaviour', 'behaviours', 'mental health', 'mental illness', 'mental disorder', 'psychiatry', 'psychoanalysis', 'psychoanalytic', 'therapist', 'therapy', 'therapies', 'clinical psychology', 'abnormal psychology', 'positive psychology', 'forensic psychology', 'child psychology', 'adolescent psychology', 'sports psychology', 'educational psychology', 'industrial-organizational psychology', 'social psychology'],
    'Business': ['management', 'marketing', 'finance', 'accounting', 'economics', 'entrepreneurship', 'strategy', 'operations', 'leadership', 'resources', 'bank', 'credit', 'money', 'market', 'stock', 'stocks', 'investment', 'investments', 'economy', 'economies', 'economic', 'economics', 'financial', 'finances', 'accounting', 'account', 'accounts', 'accountant', 'accountants', 'entrepreneur', 'entrepreneurs', 'entrepreneurship','insurance', 'insurances', 'insurer', 'insurers', 'insure', 'insured', 'insuring', 'insures', 'invest', 'invested'],
    'Gender': ['feminism', 'queer', 'intersectionality', 'masculinity', 'sexuality', 'gender', 'sexism', 'homophobia', 'transphobia', 'identity', 'patriarchy'],
    'Philosophy&Ethics': ['logic', 'metaphysics', 'epistemology', 'ethics', 'aesthetics', 'existentialism', 'philosophy', 'social', 'mind', 'ontology', 'deontology', 'utilitarianism', 'virtue', 'morality', 'subjectivity', 'objectivity', 'rationality', 'reasoning', 'argument', 'justification', 'dialectics', 'phenomenology', 'hermeneutics', 'postmodernism', 'structuralism', 'postcolonial', 'anarchism', 'communism', 'libertarianism', 'existentialist', 'skepticism', 'socratic', 'platonism', 'aristotelian', 'nihilism', 'humanism', 'transhumanism', 'naturalism', 'pragmatism', 'neo-kantian', 'neo-hegelian', 'critical theory', 'continental philosophy', 'analytic philosophy', 'phenomenology', 'pragmatism', 'post-structuralism', 'postmodernism', 'existentialist philosophy', 'ontology', 'deontology', 'epistemic', 'epistemology', 'hermeneutics', 'historiography', 'phenomenology', 'philosophy of language', 'philosophy of law', 'philosophy of mind', 'philosophy of religion', 'philosophy of science', 'political philosophy', 'social philosophy', 'philosophy of technology', 'existential philosophy'],
    'Politics&Society': ['democracy', 'globalization', 'human rights', 'environmental policy', 'public policy', 'political theory', 'international relations', 'race', 'class', 'gender', 'civil', 'war', 'protest', 'country', 'police'],
    'Arts': ['painting', 'sculpture', 'photography', 'music', 'film', 'theater', 'literature', 'performance', 'installation', 'design', 'contemporary', 'modern', 'color', 'fashion'],
    'Astronomy': ['planets', 'stars', 'galaxies', 'cosmology', 'astrophysics', 'astronomy', 'exoplanets', 'astrobiology', 'gravity', 'black holes', 'nebulae', 'supernovae', 'cosmic rays', 'dark matter', 'dark energy', 'telescopes', 'observatories', 'interstellar', 'intergalactic', 'red giants', 'white dwarfs', 'black dwarfs', 'neutron stars', 'pulsars', 'quasars', 'cosmic microwave background', 'cosmic inflation', 'cosmic web', 'gravitational waves', 'interplanetary', 'solar system', 'orbital mechanics', 'celestial mechanics', 'asteroids', 'comets', 'meteoroids', 'meteorites', 'moon', 'lunar', 'solar', 'eclipse', 'zodiac', 'constellations', 'Milky Way', 'Andromeda', 'Hubble', 'Kepler', 'Chandra', 'Spitzer', 'James Webb', 'planetarium', 'star chart', 'cosmic evolution', 'cosmic abundance', 'exoplanet discovery', 'extraterrestrial life', 'SETI'],
    'Literature': ['poetry', 'prose', 'fiction', 'nonfiction', 'drama', 'criticism', 'literary', 'postcolonial', 'novel', 'literature', 'film', 'writing', 'reading', 'book', 'shakespeare']
}

# Group the courses by topic and add them to a dictionary
course_groups = {
    'ComputerScience': ['Algorithms', 'VRdevelopment', 'ComputerScience'],
    'Chemistry': ['Chemistry', 'PhysicalChemistry'],
    'Physics': ['Physics'],
    'Math&Statistics': ['Data Analytics', 'QuantitativeAnalysis', 'LinearAlgebra'],
    'Pharma' : ['Pharmacology'],
    'Biology': ['Biology', 'Genetics', 'DrugBiology', 'Neuroscience', 'Phisiology'],
    'Psychology': ['Psychology', 'IntroductionToPsychology'],
    'Business': ['Business', 'Marketing', 'Management'],
    'Gender': ['IntroGenderSexuality', 'GenderSexuality'],
    'Philosophy&Ethics' : ['Philosophy', 'Ethics', 'IntroductionToEthics'],
    'Politics&Society': ['Social Politics', 'LatinAmericanGovPolitics', 'Politics', 'InternationalSocialJustice', 'Race&Racism'],
    'Arts': ['Art'],
    'Astronomy': ['Astronomy'],
    'Literature': ['Literature', 'ReadingLiterature', 'ReadingFilm'],
}

def get_vector_centroid(crds1):
    '''
    Find the centroid of a set of vectors.

    @param crds: A matrix containing all of the vectors.

    @return: The centroid of the rows of the matrix crds.
    '''
    centroid1 = np.zeros(300)

    for i in range(len(crds1)):
        centroid1 += crds1[i]

    centroid1 /= float(len(crds1))

    return centroid1 

def text2emb(words, model):
    return [model[word] for word in words if word in model]

def nearest_cat(refs, word, model):
    
    emb = model[word]
    d_min = 9999
    
    for cat in refs.keys():
        
        #d = dot(emb, refs[cat])/(norm(emb)*norm(refs[cat]))
        d = distance.euclidean(emb, refs[cat])
        #d = np.linalg.norm(emb-refs[cat]) # minus eucl
        if (d < d_min):

            d_min = d
            final_cat = cat
    
    return final_cat, d_min

def get_k_nearest_words_from_cat(centroids, model, k=10):

    nearest_words = dict()
    for cat in centroids.keys():
        nearest_words[cat] = []
    for word in tqdm(model.keys()):
        for cat in centroids.keys():
            d = distance.euclidean(model[word], centroids[cat])
            nearest_words[cat].append((word, d))
            if len(nearest_words[cat]) > k:
                nearest_words[cat].sort(key=lambda tup: tup[1])
                nearest_words[cat].pop()
    return nearest_words

#Function for basic cleaning/preprocessing texts
def clean(doc):
    # Removal of punctuation marks (.,/\][{} etc) and numbers
    doc = "".join([char for char in doc if char not in string.punctuation and not char.isdigit()])
    # Removal of accents
    doc = doc.encode('ascii', 'ignore').decode('ascii')
    # Removal of whitespaces
    doc = " ".join(doc.split())
    # Lowercasing
    return doc.lower()

def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        nouns = []
        for word,pos in nltk.pos_tag(nltk.word_tokenize(str(sentence))):
            #if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
            nouns.append(word)
        yield(nouns)

def remove_stopwords(texts, stop_words):
    return [[word for word in simple_preprocess(str(doc)) 
             if word not in stop_words] for doc in texts]
# Group courses into topics
def group_courses(course):
    for key, value in course_groups.items():
        if course in value:
            return key
    return np.nan

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def get_stopwords():
    nltk.download('stopwords')
    return stopwords.words('english')

def get_topic_keywords_embed(model):
    # Keywords Embeddings
    topic_keywords_embed = dict()
    for k,l in topic_keywords.items():
        topic_keywords_embed[k] = []
        for e in l:
            if e in model:
                topic_keywords_embed[k].append(model[e])
    return topic_keywords_embed

def get_topic_centroids(topic_keywords_embed):
    topic_centroids = dict()
    for cat in topic_keywords_embed.keys():
        topic_centroids[cat] = get_vector_centroid(topic_keywords_embed[cat])
    return topic_centroids

def sentence_nearest_cat(words, topic_centroids, model):
    emb = text2emb(words, model)
    cent = get_vector_centroid(emb)
    d_min = 9999

    for cat in topic_centroids.keys():
        d = distance.euclidean(cent, topic_centroids[cat])
        if (d < d_min):
            d_min = d
            final_cat = cat
    
    return final_cat, d_min

def sentence_cats_probs(words, topic_centroids, model, topic_to_idx):
    emb = text2emb(words, model)
    cent = get_vector_centroid(emb)

    # list of len number of categories
    result = [0] * len(topic_centroids.keys())

    for cat in topic_centroids.keys():
        d = distance.euclidean(cent, topic_centroids[cat])
        result[topic_to_idx[cat]] = -d
    # softmax
    result = np.exp(result) / np.sum(np.exp(result), axis=0)
    
    return result