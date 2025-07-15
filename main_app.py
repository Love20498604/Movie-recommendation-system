import pandas as pd
import numpy as np
import nltk 
from flask import Flask , request , render_template
import requests
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
import ast  #using this library to take a list from a string format to perform some operations later
import re     # use to clear punctuations



app = Flask(__name__)





def recommend_movie(movie):


    movie=movie.lower()
    movie_index=new_movies_data_df[new_movies_data_df['title']== movie].index[0]

    similarites_list= all_similarity_pairs_matrix[movie_index]  # a Row   #we will get a list of tuple where (  index , similarity ) are paired up ..
    movie_list= sorted(similarites_list , reverse=True , key=lambda x:x[1] )[1:6]   # we have sorted similarity list in descending order on the basis of similarities score
   
    recommend_movies=[]
    # print(f"Top 5 movies similar to '{movie}':")
    for each_movie in movie_list:
        title=new_movies_data_df.iloc[each_movie[0]].title
        movie_id=int( new_movies_data_df.iloc[each_movie[0]].movie_id)
        # print(recommend)
        recommend_movies.append(  (title , movie_id)   ) 
    return recommend_movies


def extract_poster( movie_id):

    #this is my api key for the extracting the poster using moive id 
    api_key="bf0711ecf17569f2ea8e59265cab57c4"
    image_url=f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
    
    images_resposne = requests.get(image_url)
    image_data=images_resposne.json()
    poster_path = image_data.get("poster_path")

    if poster_path:
        return f"https://image.tmdb.org/t/p/w500{poster_path}"
    else: 
        return None



@app.route('/' , methods=['GET'])
def index():
    all_movies = new_movies_data_df['title'].tolist()
    return render_template('index.html' , recommendations=None , all_movies=all_movies)


@app.route('/recommend' , methods=['POST'])
def recommend():

    movie = request.form['movie']
    try:
        result=[]
        recommendations= recommend_movie(movie)
        for title , movie_id in recommendations:
            poster=extract_poster(movie_id)
            result.append( {'title': title, 'poster': poster})
    except:
        recommendations="Movie Not Found in our dataset"

    all_movies = new_movies_data_df['title'].tolist()
    return render_template('index.html', recommendations=result, all_movies=all_movies)


movie_df = pd.read_csv("movies.csv")
# print(movie_df.shape)

credit_df= pd .read_csv("credits.csv")
# print(credit_df.head())

movie_df=movie_df.merge(credit_df , on='title')   # we merged both data frames into ONE to avoid working on two different dataframe.


# now we will take important coloums so that we can merge their words all together and convert
#  into a new coloumn . which would help us to convert into vector and use cosine similarity .

# print(movie_df.info())

#clean the data but choose important coloumns

movie_df = movie_df[  [ 'movie_id' , 'genres' , 'keywords' , "original_title" , 'overview' , 'title' , 'cast' , 'crew']]

# print(movie_df.info())

#now by info() i can see some of rows are missing some data and we can find out .isnull().sum()

# print(movie_df.isnull().sum())


#we can get rid of those rows by dropna()

movie_df.dropna(inplace=True)
# print(movie_df.isnull().sum())


# we need to make sure that no row is repeating itself
# print(movie_df.duplicated().sum())


#converting the data of genres coloumn of a row into a right format 
genres_data= movie_df.iloc[0].genres
# print(type(genres_data) , genres_data)

def mod_data(genres_data):

    final_list=[]
    for dict in ast.literal_eval(genres_data):
        final_list.append(dict['name'])
    return final_list




movie_df['genres']=movie_df['genres'].apply(mod_data)    # this would change the nentire coloumns of genres

# print(movie_df['genres'])


keyword_data=movie_df.iloc[0].keywords
# print(keyword_data)

movie_df['keywords']=movie_df['keywords'].apply(mod_data)

# print(movie_df[  [  'keywords' , 'genres']])
# now our data from keywords and genres are in right format , list of words


cast_data= movie_df.iloc[0].cast


def mod_cast_data(cast_data):
    final_list=[]
    number=1
    for dict in ast.literal_eval(cast_data):  # since cast_data is also a string , we need to convert into a list first using this
        if number <=3:

            final_list.append(dict['name'])
            number+=1
        else:
            break

    return final_list



movie_df['cast']= movie_df['cast'].apply(mod_cast_data)



# print(movie_df.isnull().sum())

# print(movie_df['crew'][0]) 
# now we need only name of director in put crew coloumn 


crew_data = movie_df['crew'][0]
def mod_crew_data(crew_data):
    final_list=[]
    for dict in ast.literal_eval(crew_data):
        if dict['job']=='Director':
            final_list.append(dict['name'])
    return final_list

movie_df['crew']= movie_df['crew'].apply(mod_crew_data)

# print(movie_df[ ['title' , 'crew' , 'genres' , 'keywords']])

overview_data = movie_df['overview'][0]
# print(overview_data)

def mod_overview_data(overview_data):
    return overview_data.split()
# print(mod_overview_data(overview_data))

movie_df['overview'] = movie_df['overview'].apply(mod_overview_data)

#now we have chnaged the data from all coloumns into list 

#now we need to remove space from all words which are in one word format word

def remove_spaces(data):
    final_list=[]
    for word in data:
        cleaned_word=word.replace(" " , "" )
        final_list.append(cleaned_word)
    return final_list

# data= movie_df['keywords'][0]
# print(data)
# print(remove_spaces(data))

#we can apply this function to all of the coloumns

# print(movie_df.info())   # genres , keywords, overview , cast , crew

movie_df['cast']=movie_df['cast'].apply(remove_spaces)
movie_df['genres']=movie_df['genres'].apply(remove_spaces)
movie_df['overview']=movie_df['overview'].apply(remove_spaces)
movie_df['crew']=movie_df['crew'].apply(remove_spaces)

# print(movie_df[  [ 'cast' , 'genres']])
# print(movie_df[  [ 'keywords' , 'genres']])

#creating a new coloumn with data from different coloumns

movie_df['tags']= movie_df['overview'] +  movie_df['genres'] + movie_df['keywords'] +  movie_df['cast'] + movie_df['crew']

# print(movie_df.head())


new_movies_data_df= movie_df[ ['movie_id', 'title' , 'tags']]

# print(new_movies_data_df.head())

#now we have created a new df from specific colounms and tag coloumn is written in a list but i need it in a string

new_movies_data_df['tags'] = new_movies_data_df['tags'].apply(lambda x:" ".join(x))
#converting all the words into a lower case to avoid confusion and better readiblity
new_movies_data_df['tags']=new_movies_data_df['tags'].apply(lambda x:x.lower())
new_movies_data_df['title']=new_movies_data_df["title"].apply(lambda x:x.lower())
# i have noticed one more problem in the data where i see similar words but with punctuation with it like [ king , king,  , king. ]-- words has these punctuation added.
# in order to remove them , we need to clean it up by using re (library)
def remove_punctuations( text):
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()
new_movies_data_df['tags']=new_movies_data_df["tags"].apply(remove_punctuations)


#one of the problem i see now , there are words which are same but in the diffrent form like love --> loving , loved , love
# in order to solve it , we need the stemming process which will change it same word --> love
# we are going to use library for this --> NLTK ==
#we also need to stem the data , so that different form of words stays the same , we dont need their differnt forms of a single word
def stem_words(text):
    return " ".join([ps.stem(word) for word in text.split()])

new_movies_data_df['tags'] = new_movies_data_df['tags'].apply(stem_words)

# print(new_movies_data_df['tags'][0])

# now data is ready to convert into vector ..

# merge all data from all tags into a big box called vocabulary

big_data = new_movies_data_df['tags']

# print(big_data)


# we wont add stop words like ( is , am , are ) / common words in our all collection 


stop_words = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "doing",
    "a", "an", "the",
    "and", "but", "if", "or", "because", "as", "until", "while",
    "of", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very",
    "can", "will", "just", "don", "should", "now",
    "d", "ll", "m", "o", "re", "s", "t", "ve", "y",
    "ain", "aren", "couldn", "didn", "doesn", "hadn",
    "hasn", "haven", "isn", "ma", "mightn", "mustn",
    "needn", "shan", "shouldn", "wasn", "weren", "won",
    "wouldn" , '-' , 'who'
]
)

all_words=[]
for each_row in big_data:
    each_word=each_row.split()
    for word in each_word:
        if word not in stop_words:
            all_words.append(word)

# print( len(all_words))


#now we will see the every word occurence and take it a dictionary

word_freq={}
for word in all_words:
    if word not in  word_freq:
        word_freq[word]=1
    else:
        word_freq[word]+=1

word_freq_list=[]
for word in word_freq:
    word_freq_list.append( (word , word_freq[word]) )

#now its time to sort this dictionary in descending order
#using bubble sort
# for i in range(len(word_freq_list)):

#     for j in range(i + 1, len(word_freq_list)):

#         if word_freq_list[j][1] > word_freq_list[i][1]:
#             word_freq_list[i], word_freq_list[j] = word_freq_list[j], word_freq_list[i]
word_freq_list = sorted(word_freq_list, key=lambda x: x[1], reverse=True)

new_data= word_freq_list[:5000]

# print(new_data)

# we dont need to keep the number of occurences now , instead we will take out top 5000  words and assign them an index since they are already sorted in descening order on the basis of their number of occurences.

word_with_index={}
counter=0
for word_pair in new_data:
    word_with_index[word_pair[0]]=counter
    counter+=1

# print(word_with_index)
# i am trying to sort them by alphabettically , so that i can see about stemming actually worked or not .

# since the sorting deos not work on the dict , i woudl need to extact words wothout their index into a list
#incase we want to take a look at the data

"""list_of_words=[]
for key in word_with_index:
    list_of_words.append( key)

for i  in range(len(list_of_words)):
    for j in range( len(list_of_words)-i-1):
        if list_of_words[j]> list_of_words[j+1]:
            list_of_words[j] , list_of_words[j+1] = list_of_words[j+1] , list_of_words[j]


print(list_of_words)
"""
#now we dictionary with each word with thier index listed storeed into word_with_index

#we need to compare with each row and convert each row into a vector

# print(new_movies_data_df["tags"][0])


vectors=[]
for each_row in new_movies_data_df["tags"]:
    vector_row=[0] * 5000
    list_words_in_row=each_row.split()
    
    for word in list_words_in_row:
        if word in word_with_index:
           index= word_with_index[word]    # by assinging the key into a dict we can find the value of index and them increase by 1 into a vector row1
           vector_row[index]+=1
    vectors.append(vector_row)
    

# print(len(vectors[0]))
  
from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# vectors_np=np.array(vectors)  # we are converting our array matrix into numpy array format
similarity_matrix=cosine_similarity(vectors)

#since we need our matrix in the pair of their fixed index and similarity , so that we woont lose the index #
#using enumerate built in functin , it pairs up the each similarity with its index

all_similarity_pairs_matrix = []

for i in range(len(similarity_matrix)):

    similarity_row = similarity_matrix[i]
    similarity_pairs = []

    for j, score in enumerate(similarity_row):
        similarity_pairs.append((j, float(score)))  # convert np.float64 to float
    all_similarity_pairs_matrix.append(similarity_pairs)

# print(all_similarity_pairs[0])

# this function will help us in recomending a number of similar movies by 
# first we neeed to find the index of movie from our dataset
# then once we find the index , we can find its row from similarity matrix 
#then we can sort the similarity row vectors and find out the top 5 similar movies 
#but there is a problem , we would loose the index after sorting 
#solution --> we can just pair this list of vectors wiht their index like [ (0, 1) , (1 , 0.63 ) , ( 2 , 0.09)] in whichh pair[0] represent = index and pair [1] represent calculated similarity.




if __name__=="__main__" :
    app.run(debug=True)
