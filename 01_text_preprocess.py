train['review'] = train['review'].replace('\n','').replace('\r','')
train['replyContent'] = train['replyContent'].replace('\n','').replace('\r','')
test['review'] = test['review'].replace('\n','').replace('\r','')
test['replyContent'] = test['replyContent'].replace('\n','').replace('\r','')
