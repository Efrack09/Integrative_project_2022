import tweepy
import credentials
import pymongo
import re
from textblob import TextBlob
import json

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
class TweetPrinterV2(tweepy.StreamingClient):

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
        stopwords = ["s","for", "on", "an", "a", "of", "and", "in", "the", "to", "from", "co", "t", "no", "yes"]
        temp = tweet.lower()
        temp = re.sub("'", "", temp) # to avoid removing contractions in english
        temp = re.sub("@[A-Za-z0-9_]+","", temp)
        temp = re.sub("#[A-Za-z0-9_]+","", temp)
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

        device = tweet

        if str(device) == 'Twitter for iPhone':
            cIphone += 1
        elif str(device) == 'Twitter for Android':
            cAndroid += 1
        elif str(device) == 'Twitter Web App':
            cWeb += 1
        else:
            cOthers += 1
            

    def on_data(self, tweet):

        global contador

        contador += 1


        dTweet = json.loads(tweet.decode('unicode_escape'))


        print(dTweet)
        print(dTweet['data']['created_at'])

        tweetClean = self.clean_tweet(dTweet['data']['text'])
        test =  self.get_device(dTweet['data']['source'])
        
        tweetsRT = {}

        tweetsRT['id'] =  dTweet['data']['id']
        #tweetsRT['entities'] =  tweet.entities
        tweetsRT['source'] =  dTweet['data']['source']
        tweetsRT['lang'] =  dTweet['data']['lang']
        #tweetsRT['public_metrics'] =  tweet.public_metrics
        #tweetsRT['created_at'] = tweet.created_at 
        tweetsRT['text'] =   dTweet['data']['text']
        #tweetsRT['geo'] =  tweet.geo
        #tweetsRT['hora'] =  str(dTweet['data']['created_at']).split(' ')[1]
        #tweetsRT['fecha'] = str(dTweet['data']['created_at']).split(' ')[0]
        tweetsRT['totalTweets'] = contador
        tweetsRT['cleanTweet'] = tweetClean
        tweetsRT['Iphone'] = cIphone
        tweetsRT['Android'] = cAndroid    
        tweetsRT['Web'] = cWeb   
        tweetsRT['Others'] = cOthers  
        tweetsRT['hashtag'] = hashtag 
        #mycol.insert_one(tweetsRT)

        
        print(tweetsRT)





        return True

    def on_connect(self):
      print('Connected..!')
        
    def on_error(self, status):
      print(status)
      return True


stream = TweetPrinterV2(credentials.BEARER_TOKEN)

rule_ids = []
result = stream.get_rules()
for rule in result.data:
    print(f"rule marked to delete: {rule.id} - {rule.value}")
    rule_ids.append(rule.id)
 
if(len(rule_ids) > 0):
    stream.delete_rules(rule_ids)
    printer = TweetPrinterV2(credentials.BEARER_TOKEN)
else:
    print("No rules to delete")
global hashtag 
hashtag = 'messi'
rule = tweepy.StreamRule(value=f'{hashtag} -is:retweet has:geo')
stream.add_rules(rule)
stream.filter(tweet_fields=["geo","created_at","author_id", 'public_metrics', 'organic_metrics', 'source', 'lang'],place_fields=["id","geo","name","country_code","place_type","full_name","country"],expansions=["geo.place_id", "author_id"])