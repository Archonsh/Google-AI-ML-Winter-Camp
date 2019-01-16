# encoding=utf-8
import jieba
import pandas as pd
import re

raw_data = pd.read_csv("DMSC.csv")
df = pd.DataFrame(raw_data)

name_dic = {'复仇者联盟2':'2015-05-12', '大鱼海棠':"2016-07-08", '美国队长3':"2016-05-06", '十二生肖':"2012-12-20",
            '九层妖塔':"2015-09-30", '大圣归来':"2015-07-10", '栀子花开':"2015-07-10", '夏洛特烦恼':"2015-09-30",
            '钢铁侠1':"2008-04-30", '西游降魔篇':"2013-02-10", '西游伏妖篇':"2017-01-28", '爱乐之城':"2017-02-14",
            '泰囧':"2012-12-12", '何以笙箫默':"2015-04-30", '湄公河行动':"2016-09-30", '七月与安生':"2016-09-14",
            '复仇者联盟':"2012-05-05", '后会无期':"2014-07-24", '寻龙诀':"2015-12-18", '长城':"2016-12-16",
            '左耳':"2015-04-24", '美人鱼':"2016-02-08", '小时代1':"2013-06-27", '小时代3':"2014-07-17",
            '釜山行':"2016-07-20",'变形金刚4':"2014-06-27", '你的名字':"2016-12-02", '疯狂动物城':"2016-03-04"}

df['time_interval'] = df.apply(lambda x: (pd.to_datetime(x['Date']) - pd.to_datetime(name_dic[x['Movie_Name_CN']])).days, axis=1)

stopwords = [line.strip() for line in open("stopword.txt", 'r').readlines()]

def func(row):
    tmp_list = jieba.cut(row['Comment'], cut_all = False)
    mark_count1 = 0 #! ~
    mark_count2 = 0 #?
    mark_count3 = 0 #...
    word_count = 0
    outstr = ""
    for word in tmp_list:
        if (word == '!' or word == '~' or word == '！' or word =="√" or word == "★" or word =="～"):
            mark_count1 += 1
        if (word == '?'or word == '？'):
            mark_count2 += 1
        if (word == '...' or word == '......' or word == '。。。'):
            mark_count3 += 1
        if word not in stopwords:
            word_count += 1
            outstr += word + " "
    return pd.Series([outstr, mark_count1, mark_count2, mark_count3, word_count])
tmp = pd.DataFrame()
tmp = df.apply(func, axis=1)
df['new_comment'] = tmp[0]
df['mark!'] = tmp[1]
df['mark?'] = tmp[2]
df['mark...'] = tmp[3]
df['word_num'] = tmp[4]

df.drop('Comment',axis=1, inplace=True)
df.rename(columns={'new_comment':'Comment'}, inplace = True)

columns = ['ID', 'Movie_Name_EN', 'Movie_Name_CN', 'Crawl_Date', 'Number',
           'Username', 'Date', 'Star', 'Comment', 'Like', 'time_interval',
          'mark!','mark?','mark...','word_num']

df.to_csv("data.csv",encoding="utf_8_sig",index=False,columns=columns)


