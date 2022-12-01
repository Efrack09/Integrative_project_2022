import tweepy
import credentials
import pymongo
import re
from textblob import TextBlob
import classificador


bearer_token = credentials.BEARER_TOKEN
#connecting with the database sample_airbnb

myclient = pymongo.MongoClient("mongodb+srv://ja378339:socrates314@cluster0.8srrj.mongodb.net/?retryWrites=true&w=majority")
mydb = myclient["bigdata"]
#getting the collection
mycol = mydb["juevesFinal"]



contador = 0
cPositive = 0
cNeutral = 0
cNegative = 0

cIphone = 0
cAndroid = 0
cWeb = 0
cOthers = 0


class TweetPrinterV2(tweepy.StreamingClient):

    def classTweet(self, tweet):
        classi = classificador.predict_new(tweet.text)
        print(classi)

    def contar():
        global counterT
        counterT = counterT + 1
        print("contar() ha sido llamado " + str(counterT) + " veces")  


    def clean_tweet(self, tweet):
        '''
        Utility function to clean tweet text by removing links, special characters
        using simple regex statements.
        '''
        '''
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())
        '''
        stopwords = ["s", "an", "a", "co", "t", "c", "l", "un", "y", "ma", "la", "d", "que", "por", "el", "n", "lo", "para", "n", "Tco ", "Vwtpsx", "Agv", 
        "Bd"]
        temp = tweet.lower()
        temp = re.sub("'", "", temp) # to avoid removing contractions in english
        temp = re.sub("@[A-Za-z0-9_]+","", temp)
        #temp = re.sub("#[A-Za-z0-9_]+","", temp)
        temp = re.sub(r'http\S+', '', temp)
        temp = re.sub('[()!?]', ' ', temp)
        temp = re.sub('\[.*?\]',' ', temp)
        temp = re.sub("[^a-z0-9]"," ", temp)
        temp = temp.split()
        temp = [w for w in temp if not w in stopwords]
        temp = " ".join(word for word in temp)
        return temp    
    

    def get_device(self, tweet):
        '''
        Utility function to classify sentiment of passed tweet
        using textblob's sentiment method
        '''
        global cIphone  
        global cAndroid 
        global cWeb 
        global cOthers 
        # create TextBlob object of passed tweet text
        device = tweet
        # set sentiment
        #for i in listSource:
        if str(device) == 'Twitter for iPhone':
            cIphone += 1
        elif str(device) == 'Twitter for Android':
            cAndroid += 1
        elif str(device) == 'Twitter Web App':
            cWeb += 1
        else:
            cOthers += 1

    def getAllinfo(self, tweet):

        global contador

        contador += 1
    
        tweetClean = self.clean_tweet(tweet.text)
        test =  self.get_device(tweet.source)


        tweetsRT = {}

        tweetsRT['id'] =  tweet.id
        #tweetsRT['entities'] =  tweet.entities
        tweetsRT['source'] =  tweet.source
        tweetsRT['lang'] =  tweet.lang
        #tweetsRT['public_metrics'] =  tweet.public_metrics
        #tweetsRT['created_at'] = tweet.created_at 
        tweetsRT['text'] =  tweet.text
        #tweetsRT['geo'] =  tweet.geo
        tweetsRT['hora'] =  str(tweet.created_at).split(' ')[1]
        tweetsRT['fecha'] = str(tweet.created_at).split(' ')[0]
        tweetsRT['totalTweets'] = contador
        tweetsRT['cleanTweet'] = tweetClean
        tweetsRT['Iphone'] = cIphone
        tweetsRT['Android'] = cAndroid    
        tweetsRT['hashtagOne'] = hashtagOne 
        tweetsRT['hashtagTwo'] = hashtagTwo 
        mycol.insert_one(tweetsRT)

        #print(xd)cle 
        
        #print(f'{tweetsRT}')

        #print(f'Positives: {cPositive}, Neutral: {cNeutral}, Negative: {cNegative}')
        #print(f'Iphone: {cIphone}, Android: {cAndroid}, Web: {cWeb}, Others: {cOthers}')
        #print("TweetPrinterV2() ha sido llamado " + str(contador) + " veces") 
        print("-"*50)


    def on_tweet(self, tweet):
    
        final = self.getAllinfo(tweet)
    
        
    def on_connect(self):
        print('Connected..!')

    def on_error(self, status):
        print(status)
        return True
    

printer = TweetPrinterV2(bearer_token)
 
# clean-up pre-existing rules
rule_ids = []
result = printer.get_rules()
for rule in result.data:
    print(f"rule marked to delete: {rule.id} - {rule.value}")
    rule_ids.append(rule.id)
 
if(len(rule_ids) > 0):
    printer.delete_rules(rule_ids)
    printer = TweetPrinterV2(bearer_token)
else:
    print("No rules to delete")
 
# add new rules    
#rule = StreamRule(value="adrianaTest")
global hashtagOne 
global hashtagTwo 
hashtagOne = '#VamosArgentina'
hashtagTwo = '#VamosMexico'
# Value is the keyword, hashtag or something that we want to search
rule = tweepy.StreamRule(value=f"{hashtagOne} OR {hashtagTwo} lang:en")
printer.add_rules(rule)
printer.filter(expansions=['geo.place_id', 'author_id',],tweet_fields=['created_at', 'geo', 'entities', 'public_metrics', 'organic_metrics', 'source', 'lang'], user_fields = ['name'] )
#printer.filter(tweet_fields=["geo","created_at","author_id", 'entities', 'public_metrics', 'organic_metrics', 'source', 'lang'],place_fields=["id","geo","name","country_code","place_type","full_name","country"],expansions=["geo.place_id", "author_id"])

















