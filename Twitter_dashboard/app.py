#ion. generate_random_data() yield values from 0 to 100 and the current timestamp.

import json
import random
import time
import numpy as np
from datetime import datetime
import pymongo
import pandas as pd
import io
from wordcloud import WordCloud
import base64

from flask import Flask, Response, render_template, stream_with_context

application = Flask(__name__)
random.seed()  # Initialize the random number generator
myclient = pymongo.MongoClient("mongodb+srv://ja378339:socrates314@cluster0.8srrj.mongodb.net/?retryWrites=true&w=majority")
mydb = myclient["bigdata"]
#getting the collection 
mycol = mydb["juevesFinal"]
temp = [x for x in mycol.find().limit(1).sort([('_id',-1)])]


@application.route('/')
def index():
    return render_template('index.html')

@application.route('/chart-data')
def chart_data():
    def generate_random_data():
        while True:
            #mycol = mydb["twitterFinal"]
            #temp = [x for x in mycol.find()][-1]
            #temp = [x for x in mycol.find().limit(1).sort([('_id',-1)])]
            temp = [x for x in mycol.find().limit(1).sort([('_id',-1)])]

            json_data = json.dumps(
                {'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'value':temp[0]['totalTweets']})
            yield f"data:{json_data}\n\n"
            time.sleep(1)

    response = Response(stream_with_context(generate_random_data()), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response




    
@application.route('/mobil-data')
def mob_data():
    def get_mob_dat():
        while True:
            #mycol = mydb["twitterFinal"]
            #temp = [x for x in mycol.find()][-1]
            temp = [x for x in mycol.find().limit(1).sort([('_id',-1)])]

            json_data = json.dumps(
                {'Iphone':temp[0]['Iphone'],'Android':temp[0]['Android'],'Web':temp[0]['Web'],'Others':temp[0]['Others']})
            yield f"data:{json_data}\n\n"
            time.sleep(1)

    response = Response(stream_with_context(get_mob_dat()), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response


@application.route('/total-tweets')
def tweeT():
    def totalTweet():
        while True:
            #mycol = mydb["twitterFinal"]
            #temp = [x for x in mycol.find()][-1]
            temp = [x for x in mycol.find().limit(1).sort([('_id',-1)])]

            json_data = json.dumps(
                {'tweetsT':temp[0]['totalTweets']})
            yield f"data:{json_data}\n\n"
            time.sleep(1)

    response = Response(stream_with_context(totalTweet()), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response

@application.route('/text-tweet')
def tweeText():
    def textT():
        while True:
            #mycol = mydb["twitterFinal"]
            #temp = [x for x in mycol.find()][-1]
            temp = [x for x in mycol.find().limit(1).sort([('_id',-1)])]

            json_data = json.dumps(
                {'text':temp[0]['text']})
            yield f"data:{json_data}\n\n"
            time.sleep(1)

    response = Response(stream_with_context(textT()), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response

@application.route('/t-hashtag')
def hashtag():
    def hashtagS():
        while True:
            #mycol = mydb["twitterFinal"]
            #temp = [x for x in mycol.find()][-1]
            temp = [x for x in mycol.find().limit(1).sort([('_id',-1)])]

            json_data = json.dumps(
                {'hashtagOne':temp[0]['hashtagOne'],'hashtagTwo':temp[0]['hashtagTwo'] })
            yield f"data:{json_data}\n\n"
            time.sleep(2)

    response = Response(stream_with_context(hashtagS()), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response

@application.route('/w_wordsA')
def send_wcA():
    def get_wordcloudA():
        while True:
            key = [x for x in mycol.find().limit(1).sort([('_id',-1)])][0]['hashtagOne']

            df = pd.DataFrame([x for x in mycol.find({'hashtagOne':key})])
            #wordcount = {}
            text1 = list(df['cleanTweet'])

            text = ''
            for i in text1:
                text+= i
                    #print(i)
            path = r'C:\Users\EFRACK\Desktop\Integrative_project_2022\Twitter_dashboard\GothamRnd-Bold.otf'
            pil_img = WordCloud(colormap = 'Blues', font_path=path, mode = "RGBA", background_color = None, max_words=1500).generate(text=text).to_image()
            img = io.BytesIO()
            pil_img.save(img, "PNG")
            img.seek(0)
            img_b64 = base64.b64encode(img.getvalue()).decode()

            json_data = json.dumps(
                {'img':img_b64})
            yield f"data:{json_data}\n\n"
            time.sleep(3)

    response = Response(stream_with_context(get_wordcloudA()), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response

@application.route('/w_wordsM')
def send_wcM():
    def get_wordcloudM():
        while True:
            key = [x for x in mycol.find().limit(1).sort([('_id',-1)])][0]['hashtagTwo']

            df = pd.DataFrame([x for x in mycol.find({'hashtagTwo':key})])
            #wordcount = {}
            text1 = list(df['cleanTweet'])

            text = ''
            for i in text1:
                text+= i
                    #print(i)
            path = r'C:\Users\EFRACK\Desktop\Integrative_project_2022\Twitter_dashboard\GothamRnd-Bold.otf'
            pil_img = WordCloud(colormap = 'Greens', font_path=path, mode = "RGBA", background_color = None, max_words=1500).generate(text=text).to_image()
            img = io.BytesIO()
            pil_img.save(img, "PNG")
            img.seek(0)
            img_b64 = base64.b64encode(img.getvalue()).decode()

            json_data = json.dumps(
                {'img':img_b64})
            yield f"data:{json_data}\n\n"
            time.sleep(3)

    response = Response(stream_with_context(get_wordcloudM()), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response

@application.route("/serve_csv",methods=['GET'])
def serve_csv():
    def totalTweet():
        while True:
            #mycol = mydb["twitterFinal"]
            #temp = [x for x in mycol.find()][-1]
            #temp = [x for x in mycol.find().limit(1).sort([('_id',-1)])]
            df= pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv')
            df['data'] = np.random.randint(500, size=len(df))
 
            #json_data = json.dumps(
            #    {'tweetsT':temp[0]['totalTweets']})
            json_data = json.dumps(
                {'CODE':list(df['CODE']),'GDP': list(df['data']),'COUNTRY':list(df['COUNTRY'])})
            yield f"data:{json_data}\n\n"
            time.sleep(5)
 
    response = Response(stream_with_context(totalTweet()), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response
    

if __name__ == '__main__':
    application.run(debug=True, threaded=True)
