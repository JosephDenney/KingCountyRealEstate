# print feature lm plot
def feat_plots(feature,target,df):
    print(feature)
    
    plt.title("{} histogram".format(feature))
    sns.distplot(df[feature])
    plt.show()
    
    sns.lmplot(x=feature, y=target, data=df, line_kws={'color': 'red'})
    plt.title("{} vs {}".format(target, feature))
    plt.show()
    
    pass
    