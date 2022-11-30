import tweepy
import credentials
import pymongo
import re
from textblob import TextBlob
import json

class MyListener(tweepy.StreamingClient):
    def on_data(self, tweet):
      res = json.loads(tweet.decode('unicode_escape'))
      

      #print(json.dumps(data_obj,indent=2))
      print(res)
      print('----------------')
      print(res['data'])
      print(res['includes'])
      print(res['places'])
      return True

    def on_connect(self):
      print('Connected..!')
        
    def on_error(self, status):
      print(status)
      return True


stream = MyListener(credentials.BEARER_TOKEN)

rule_ids = []
result = stream.get_rules()
for rule in result.data:
    print(f"rule marked to delete: {rule.id} - {rule.value}")
    rule_ids.append(rule.id)
 
if(len(rule_ids) > 0):
    stream.delete_rules(rule_ids)
    printer = MyListener(credentials.BEARER_TOKEN)
else:
    print("No rules to delete")
    
rule = tweepy.StreamRule(value='messi -is:retweet has:geo')
stream.add_rules(rule)
stream.filter(tweet_fields=["geo","created_at","author_id", 'public_metrics', 'organic_metrics', 'source', 'lang'],place_fields=["id","geo","name","country_code","place_type","full_name","country"],expansions=["geo.place_id", "author_id"])