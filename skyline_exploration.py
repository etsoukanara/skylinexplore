from graphtempo import *
import pandas as pd
import itertools
import copy
import plotly.express as px
import plotly.io as pio
pio.renderers.default='svg'
import plotly.graph_objects as go
import time
import os


# plot skyline
def plot_skyline(common, event):
    data = {}
    colors = []
    x = []
    y = []
    for lst in common:
        colors.extend(['blue', 'blue', 'brown'])
        x.extend([int(lst[1][0]), int(lst[1][-1]), None])
        y.extend([str(lst[0]) + ' (PoR: ' + str(lst[-1][0]) + ')', str(lst[0]) + ' (PoR: ' + str(lst[-1][0]) + ')', None])
    data['colors'] = colors
    data['x'] = x
    data['y'] = y
    
    fig = go.Figure(
        data=[
            go.Scatter(
                x=data["x"],
                y=data["y"],
                mode="lines",
                marker=dict(
                    color="#4169E1",
                ),
            ),
            go.Scatter(
                x=data["x"],
                y=data["y"],
                mode="markers+text",
                marker=dict(
                    color="#4169E1",
                    #color=data["colors"],
                    size=15,
                ),
            ),
        ]
    )
    
    fig.update_layout(
        xaxis = dict(
            tickvals = [i for i in interval],
            title = 'Interval'
            #ticktext = [i.upper() for i in interval]
        ),
        yaxis = dict(
            title = '#Interactions'
            #ticktext = [i.upper() for i in interval]
        ),
        title=event+" skyline results for " + str(attr_values) + " interactions",
        width=1500,
        height=1200,
        showlegend=False,
        font_size=20,
        plot_bgcolor='rgba(0,0,0,0)'
    )
    fig.update_traces(marker_size=15, line=dict(width=2.5))#, line_color="#4169E1")
    fig.update_xaxes(showgrid=True, gridcolor='lightgrey', showline=True, linecolor = 'black', zeroline=False, mirror=True)
    fig.update_yaxes(showgrid=True, gridcolor='lightgrey', showline=True, linecolor = 'black', zeroline=False, mirror=True)
    
    fig.show()


# dblp dataset
stc_attrs = ['gender']
varying = ['#Publications']
edges_df = pd.read_csv('C:/Users/Lila/Desktop/PROJECT_2.1_DEMO/GraphTempo_APP/datasets/dblp_dataset/edges.csv', sep=' ', index_col=[0,1])
nodes_df = pd.read_csv('C:/Users/Lila/Desktop/PROJECT_2.1_DEMO/GraphTempo_APP/datasets/dblp_dataset/nodes.csv', sep=' ', index_col=0)
time_variant_attr = pd.read_csv('C:/Users/Lila/Desktop/PROJECT_2.1_DEMO/GraphTempo_APP/datasets/dblp_dataset/time_variant_attr.csv', sep=' ', index_col=0)
time_invariant_attr = pd.read_csv('C:/Users/Lila/Desktop/PROJECT_2.1_DEMO/GraphTempo_APP/datasets/dblp_dataset/time_invariant_attr.csv', sep=' ', index_col=0)
time_invariant_attr.rename(columns={'0': 'gender'}, inplace=True)
nodes_df.index.names = ['userID']
time_invariant_attr.gender.replace(['female','male'], ['F','M'],inplace=True)
interval = [i for i in edges_df.columns]

# movielens dataset
edges_df = pd.read_csv('C:/Users/Lila/Desktop/PROJECT_2.1_DEMO/GraphTempo_APP/datasets/movielens_dataset/edges.csv', sep=' ')
edges_df.set_index(['Left', 'Right'], inplace=True)
nodes_df = pd.read_csv('C:/Users/Lila/Desktop/PROJECT_2.1_DEMO/GraphTempo_APP/datasets/movielens_dataset/nodes.csv', sep=' ', index_col=0)
time_variant_attr = pd.read_csv('C:/Users/Lila/Desktop/PROJECT_2.1_DEMO/GraphTempo_APP/datasets/movielens_dataset/time_variant_attr.csv', sep=' ', index_col=0)
time_invariant_attr = pd.read_csv('C:/Users/Lila/Desktop/PROJECT_2.1_DEMO/GraphTempo_APP/datasets/movielens_dataset/time_invariant_attr.csv', sep=' ', index_col=0)
interval = [i for i in edges_df.columns]

# =============================================================================
# # replace notation of attributes
# time_invariant_attr.gender.replace(['F','M'], [0,1],inplace=True)
# time_invariant_attr.occupation.replace(['other', 'academic/educator', 'artist', \
#         'clerical/admin', 'college/grad student', 'customer service', \
#         'doctor/health care', 'executive/managerial', 'farmer', 'homemaker', \
#         'K-12 student', 'lawyer', 'programmer', 'retired', 'sales/marketing', \
#         'scientist', 'self-employed', 'technician/engineer', \
#         'tradesman/craftsman', 'unemployed', 'writer'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, \
#         15, 16, 17, 18, 19, 20], inplace=True)
# time_invariant_attr.age.replace(['Under 18', '18-24', \
#                     '25-34', '35-44', '45-49', '50-55', '56+'], [1, 18, 25, 35, 45, 50, 56], inplace=True)
# =============================================================================

# school dataset
edges_df = pd.read_csv('C:/Users/Lila/Desktop/PROJECT_2.2_EXT/experiments/school_dataset/edges.csv', sep=' ', index_col=[0,1])
nodes_df = pd.read_csv('C:/Users/Lila/Desktop/PROJECT_2.2_EXT/experiments/school_dataset/nodes.csv', sep=' ', index_col=0)
time_invariant_attr = pd.read_csv('C:/Users/Lila/Desktop/PROJECT_2.2_EXT/experiments/school_dataset/time_invariant_attr.csv', sep=' ', index_col=0)
nodes_df.index.names = ['userID']
interval = [i for i in edges_df.columns]

#domain
time_invariant_attr['gender'].value_counts()
time_invariant_attr['gender'].nunique()

#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#

# SKYLINE

############################## ADBIS EXPERIMENTS ##############################

x = [('F', 'F'), ('F', 'M'), ('M', 'F'), ('M', 'M')]
y = [('1A','1A'), ('1B','1B'), ('2A','2A'), ('2B','2B'), ('3A','3A'), \
     ('3B','3B'), ('4A','4A'), ('4B','4B'), ('5A','5A'), ('5B','5B')]

# Stability (tnew(inx)&told / told(inx)&tnew) (maximal)
for xi in x:
    attr_values = xi
    stc_attrs = ['gender']
    

# SKYLINE - NEW IMPLEMENTATION

# Stability (told(inx)&tnew) (maximal)

attr_values = ('F', 'F')
stc_attrs = ['gender']


def Stab_INX_MAX(attr_val,stc,nodes,edges,time_inv):
    c=0
    intvls = []
    for i in range(1,len(edges.columns)+1-c):
        intvls.append([list(edges.columns[:i]), list(edges.columns[i:i+1])])
        c += 1
    intvls = intvls[:-1]
    skyline = {i:[] for i in range(1, len(edges.columns))}
    dominate_counter = {}
    for left,right in intvls:
        max_length = len(left)
        while len(left) >= 1:
            #print(left)
            inx,tia_inx = Intersection_Static(nodes,edges,time_inv,left+right)
            if not inx[1].empty:
                agg_inx = Aggregate_Static_Dist(inx,tia_inx,stc)
                if attr_val in agg_inx[1].index:
                    current_w = agg_inx[1].loc[attr_val,:][0]
                    dominate_counter[str((current_w,left,right))] = 0
                    pr = len(left)
                    #print('pr: ', pr)
                    while not skyline[pr] and pr <= max_length:
                        pr += 1
                        #print('while..., pr: ', pr)
                        if pr > max_length:
                            break
                    if pr > max_length:
                        #print(pr, '>', max_length)
                        previous_w = 0
                    else:
                        #print(pr, '<=', max_length)
                        previous_w = skyline[pr][0][0]
                    if current_w > previous_w:
                        #print(current_w, '>', previous_w)
                        if len(left) == pr:
                            dominate_counter[str((current_w,left,right))] += 1
                            for s in skyline[pr]:
                                dominate_counter[str((current_w,left,right))] += dominate_counter[str(tuple(s))]
                            #del dominate_counter[str(tuple(s))]
                            dominate_counter[str(tuple(s))] = 0
                        skyline[len(left)] = [[current_w,left,right]]
                    elif current_w == previous_w:
                        if len(left) == pr:
                            skyline.setdefault(len(left),[]).append([current_w,left,right])
                    else:
                        #print(current_w, '<=', previous_w)
                        for s in skyline[pr]:
                            dominate_counter[str(tuple(s))] += 1
                    if len(left) > 1:
                        pr2 = len(left)-1
                        while not skyline[pr2] and pr2 >= 1:
                            pr2 -= 1
                            if pr2 == 0:
                                break
                        if pr2 > 0:
                            if skyline[pr2][0][0] <= current_w:
                                dominate_counter[str((current_w,left,right))] += 1
                                for s in skyline[pr2]:
                                    dominate_counter[str((current_w,left,right))] += dominate_counter[str(tuple(s))]
                                #del dominate_counter[str(tuple(s))]
                                dominate_counter[str(tuple(s))] = 0
                                skyline[pr2] = []
            left = left[1:]
    skyline = {i:j for i,j in skyline.items() if j}
    dominate_counter = {i:j for i,j in dominate_counter.items() \
                        if list(eval(i)) in [si for s in skyline.values() for si in s]}
    return(skyline,dominate_counter)

skyline_stab, dominance_stab = Stab_INX_MAX(attr_values,stc_attrs,nodes_df,edges_df,time_invariant_attr)
                

# growth (tnew - told(union)) (maximal)

def Growth_UN_MAX(attr_val,stc,nodes,edges,time_inv):
    c=0
    intvls = []
    for i in range(1,len(edges.columns)+1-c):
        intvls.append([list(edges.columns[:i]), list(edges.columns[i:i+1])])
        c += 1
    intvls = intvls[:-1]
    
    skyline = {i:[] for i in range(1, len(edges.columns))}
    dominate_counter = {}
    for left,right in intvls:
        max_length = len(left)
        while len(left) >= 1:
            #print(left)
            diff,tia_diff = Diff_Static(nodes,edges,time_inv,right,left)
            if not diff[1].empty:
                agg_diff = Aggregate_Static_Dist(diff,tia_diff,stc)
                if attr_val in agg_diff[1].index:
                    current_w = agg_diff[1].loc[attr_val,:][0]
                    dominate_counter[str((current_w,left,right))] = 0
                    pr = len(left)
                    #print('pr: ', pr)
                    while not skyline[pr] and pr <= max_length:
                        pr += 1
                        #print('while..., pr: ', pr)
                        if pr > max_length:
                            break
                    if pr > max_length:
                        #print(pr, '>', max_length)
                        previous_w = 0
                    else:
                        #print(pr, '<=', max_length)
                        previous_w = skyline[pr][0][0]
                    if current_w > previous_w:
                        #print(current_w, '>', previous_w)
                        if len(left) == pr:
                            dominate_counter[str((current_w,left,right))] += 1
                            for s in skyline[pr]:
                                dominate_counter[str((current_w,left,right))] += dominate_counter[str(tuple(s))]
                            dominate_counter[str(tuple(s))] = 0
                        skyline[len(left)] = [[current_w,left,right]]
                    elif current_w == previous_w:
                        if len(left) == pr:
                            skyline.setdefault(len(left),[]).append([current_w,left,right])
                    else:
                        #print(current_w, '<=', previous_w)
                        for s in skyline[pr]:
                            dominate_counter[str(tuple(s))] += 1
                    if len(left) > 1:
                        pr2 = len(left)-1
                        while not skyline[pr2] and pr2 >= 1:
                            pr2 -= 1
                            if pr2 == 0:
                                break
                        if pr2 > 0:
                            if skyline[pr2][0][0] <= current_w:
                                dominate_counter[str((current_w,left,right))] += 1
                                for s in skyline[pr2]:
                                    dominate_counter[str((current_w,left,right))] += dominate_counter[str(tuple(s))]
                                dominate_counter[str(tuple(s))] = 0
                                skyline[pr2] = []
            left = left[1:]
    skyline = {i:j for i,j in skyline.items() if j}
    dominate_counter = {i:j for i,j in dominate_counter.items() \
                        if list(eval(i)) in [si for s in skyline.values() for si in s]}
    return(skyline,dominate_counter)

skyline_grow, dominance_grow = Growth_UN_MAX(attr_values,stc_attrs,nodes_df,edges_df,time_invariant_attr)



# shrinkage (told(union) - tnew) (minimal)
def Shrink_UN_MIN(attr_val,stc,nodes,edges,time_inv):
    s = [[str(i)] for i in edges.columns[:-1]]
    e = [[str(i)] for i in edges.columns[1:]]
    intvls = list(zip(s,e))
    intvls = [list(i) for i in intvls]
    
    skyline = {i:[] for i in range(1, len(edges.columns))}
    dominate_counter = {}
    for left,right in intvls:
        min_length = len(left)
        flag = True
        while flag == True:
            #print('left',left,'right',right)
            diff,tia_diff = Diff_Static(nodes,edges,time_inv,left,right)
            if not diff[1].empty:
                #print('diff empty?', diff[1].empty)
                agg_diff = Aggregate_Static_Dist(diff,tia_diff,stc)
                if attr_val in agg_diff[1].index:
                    current_w = agg_diff[1].loc[attr_val,:][0]
                    dominate_counter[str((current_w,left,right))] = 0
                    pr = len(left)
                    #print('pr: ', pr)
                    while not skyline[pr] and pr >= min_length:
                        pr -= 1
                        #print('while..., pr: ', pr)
                        if pr < min_length:
                            break
                    if pr < min_length:
                        #print(pr, '>', max_length)
                        previous_w = 0
                    else:
                        #print(pr, '<=', max_length)
                        previous_w = skyline[pr][0][0]
                    if current_w > previous_w:
                        #print(current_w, '>', previous_w)
                        if len(left) == pr:
                            dominate_counter[str((current_w,left,right))] += 1
                            for s in skyline[pr]:
                                dominate_counter[str((current_w,left,right))] += dominate_counter[str(tuple(s))]
                            dominate_counter[str(tuple(s))] = 0
                        skyline[len(left)] = [[current_w,left,right]]
                    elif current_w == previous_w:
                        if len(left) == pr:
                            skyline.setdefault(len(left),[]).append([current_w,left,right])
                    else:
                        #print(current_w, '<=', previous_w)
                        for s in skyline[pr]:
                            dominate_counter[str(tuple(s))] += 1
                    # if left[0] not reached the end --> left != edges.columns[0]
                    if left[0] != edges.columns[0]:
                    #if left[-1] != list(edges.columns)[list(edges.columns).index(right[0])-1]:
                        pr2 = len(left)+1
                        while not skyline[pr2] and pr2 <= len(list(edges.columns)[:list(edges.columns).index(right[0])]):
                            pr2 += 1
                            if pr2 == len(list(edges.columns)[:list(edges.columns).index(right[0])])+1:
                                break
                        if pr2 < len(list(edges.columns)[:list(edges.columns).index(right[0])])+1:
                            if skyline[pr2][0][0] <= current_w:
                                dominate_counter[str((current_w,left,right))] += 1
                                for s in skyline[pr2]:
                                    dominate_counter[str((current_w,left,right))] += dominate_counter[str(tuple(s))]
                                dominate_counter[str(tuple(s))] = 0
                                skyline[pr2] = []
            if left[0] != edges.columns[0]:
                flag = True
                left = [edges.columns[list(edges.columns).index(left[0])-1]]+left
            else:
                flag = False
    skyline = {i:j for i,j in skyline.items() if j}
    dominate_counter = {i:j for i,j in dominate_counter.items() \
                        if list(eval(i)) in [si for s in skyline.values() for si in s]}
    return(skyline,dominate_counter)

skyline_shr, dominance_shr = Shrink_UN_MIN(attr_values,stc_attrs,nodes_df,edges_df,time_invariant_attr)


############################


# 1 TOP-K DOMINATING

x = [('F', 'F'), ('F', 'M'), ('M', 'F'), ('M', 'M')]
stc_attrs = ['gender']

top_stab = {}
for xi in x:
    skyline_stab, dominance_stab = Stab_INX_MAX(xi,stc_attrs,nodes_df,edges_df,time_invariant_attr)
    top_stab[str(xi)] = [skyline_stab, dominance_stab]
print('size stab', [len(i[0]) for i in top_stab.values()])

top_grow = {}
for xi in x:
    skyline_grow, dominance_grow = Growth_UN_MAX(xi,stc_attrs,nodes_df,edges_df,time_invariant_attr)
    top_grow[str(xi)] = [skyline_grow, dominance_grow]
print('size grow', [len(i[0]) for i in top_grow.values()])

top_shr = {}
for xi in x:
    skyline_shr, dominance_shr = Shrink_UN_MIN(xi,stc_attrs,nodes_df,edges_df,time_invariant_attr)
    top_shr[str(xi)] = [skyline_shr, dominance_shr]
print('size shr', [len(i[0]) for i in top_shr.values()])

# 2 TIME & SIZE EXPERIMENTS FOR DBLP

# DBLP

edges_sliced = [edges_df.iloc[:,:6], edges_df.iloc[:,:11], edges_df.iloc[:,:16], edges_df.iloc[:,:21]]

edges_sliced = [i.loc[~(i==0).all(axis=1)] for i in edges_sliced]

x = [('F', 'F'), ('F', 'M'), ('M', 'F'), ('M', 'M')]
stc_attrs = ['gender']

for xi in x:
    result = []
    for j in range(5):
        start_end_agg = []
        for y in edges_sliced:
            start = time.time()
            skyline_stab, dominance_stab = Stab_INX_MAX(xi,stc_attrs,nodes_df,y,time_invariant_attr)
            end = time.time()
            start_end_agg.append(end-start)
        result.append(start_end_agg)
    
    res = pd.DataFrame(result).T
    res[str(xi)] = res.mean(axis=1)

    # if file does not exist write header 
    if not os.path.isfile('experiments/runtime/dblp/stab_gender.csv'):
       res.to_csv('experiments/runtime/dblp/stab_gender.csv', header='column_names')
    else: # else it exists so append without writing the header
       res.to_csv('experiments/runtime/dblp/stab_gender.csv', mode='a', header='column_names')


for xi in x:
    result = []
    for j in range(5):
        start_end_agg = []
        for y in edges_sliced:
            start = time.time()
            skyline_grow, dominance_grow = Growth_UN_MAX(xi,stc_attrs,nodes_df,y,time_invariant_attr)
            end = time.time()
            start_end_agg.append(end-start)
        result.append(start_end_agg)
    
    res = pd.DataFrame(result).T
    res[str(xi)] = res.mean(axis=1)

    # if file does not exist write header 
    if not os.path.isfile('experiments/runtime/dblp/grow_gender.csv'):
       res.to_csv('experiments/runtime/dblp/grow_gender.csv', header='column_names')
    else: # else it exists so append without writing the header
       res.to_csv('experiments/runtime/dblp/grow_gender.csv', mode='a', header='column_names')


for xi in x:
    result = []
    for j in range(5):
        start_end_agg = []
        for y in edges_sliced:
            start = time.time()
            skyline_shr, dominance_shr = Shrink_UN_MIN(xi,stc_attrs,nodes_df,y,time_invariant_attr)
            end = time.time()
            start_end_agg.append(end-start)
        result.append(start_end_agg)
    
    res = pd.DataFrame(result).T
    res[str(xi)] = res.mean(axis=1)

    # if file does not exist write header 
    if not os.path.isfile('experiments/runtime/dblp/shr_gender.csv'):
       res.to_csv('experiments/runtime/dblp/shr_gender.csv', header='column_names')
    else: # else it exists so append without writing the header
       res.to_csv('experiments/runtime/dblp/shr_gender.csv', mode='a', header='column_names')


# Primary School & MovieLens for full dataset

x = [('F', 'F'), ('F', 'M'), ('M', 'F'), ('M', 'M')]
stc_attrs = ['gender']
dataset = 'primary'
dataset = 'movielens'

for xi in x:
    result = []
    for j in range(5):
        start_end_agg = []
        start = time.time()
        skyline_stab, dominance_stab = Stab_INX_MAX(xi,stc_attrs,nodes_df,edges_df,time_invariant_attr)
        end = time.time()
        start_end_agg.append(end-start)
        result.append(start_end_agg)
    
    res = pd.DataFrame(result).T
    res[str(xi)] = res.mean(axis=1)

    # if file does not exist write header 
    if not os.path.isfile('experiments/runtime/'+dataset+'/stab_gender.csv'):
       res.to_csv('experiments/runtime/'+dataset+'/stab_gender.csv', header='column_names')
    else: # else it exists so append without writing the header
       res.to_csv('experiments/runtime/'+dataset+'/stab_gender.csv', mode='a', header='column_names')


for xi in x:
    result = []
    for j in range(5):
        start_end_agg = []
        start = time.time()
        skyline_grow, dominance_grow = Growth_UN_MAX(xi,stc_attrs,nodes_df,edges_df,time_invariant_attr)
        end = time.time()
        start_end_agg.append(end-start)
        result.append(start_end_agg)
    
    res = pd.DataFrame(result).T
    res[str(xi)] = res.mean(axis=1)

    # if file does not exist write header 
    if not os.path.isfile('experiments/runtime/'+dataset+'/grow_gender.csv'):
       res.to_csv('experiments/runtime/'+dataset+'/grow_gender.csv', header='column_names')
    else: # else it exists so append without writing the header
       res.to_csv('experiments/runtime/'+dataset+'/grow_gender.csv', mode='a', header='column_names')


for xi in x:
    result = []
    for j in range(5):
        start_end_agg = []
        start = time.time()
        skyline_shr, dominance_shr = Shrink_UN_MIN(xi,stc_attrs,nodes_df,edges_df,time_invariant_attr)
        end = time.time()
        start_end_agg.append(end-start)
        result.append(start_end_agg)
    
    res = pd.DataFrame(result).T
    res[str(xi)] = res.mean(axis=1)

    # if file does not exist write header 
    if not os.path.isfile('experiments/runtime/'+dataset+'/shr_gender.csv'):
       res.to_csv('experiments/runtime/'+dataset+'/shr_gender.csv', header='column_names')
    else: # else it exists so append without writing the header
       res.to_csv('experiments/runtime/'+dataset+'/shr_gender.csv', mode='a', header='column_names')


# =============================================================================
# # plot exploration
# 
# stc_attrs = ['gender']
# #stc_attrs = ['class']
# attr_values = ('F','F')
# #attr_values = ('M','M')
# #attr_values = ('1A','1A')
# k=50
# result,myagg = Stability_Intersection_Static_a(k,interval,nodes_df,edges_df,time_invariant_attr,stc_attrs,attr_values)
# #result,myagg = Growth_Union_Static_a(k,interval,nodes_df,edges_df,time_invariant_attr,stc_attrs,attr_values)
# #result,myagg = Shrink_Union_Static_a(k,interval,nodes_df,edges_df,time_invariant_attr,stc_attrs,attr_values)
# result = result[::-1]
# 
# result_lst = []
# for lst in result:
# 	for i in lst[0]:
# 		tmp = [i,lst[1][0]]
# 		result_lst.append(tmp)
# 
# # map str to num
# str_num = {}
# for i in interval:
# 	str_num[i] = int(i)
# 
# # map num to str
# num_str = {}
# for i in interval:
# 	num_str[int(i)] = str(i)
# 
# # convert time points to strings
# result_lst = [[str_num[i],str_num[j]] for i,j in result_lst]
# 
# result_df = pd.DataFrame(result_lst)
# df_cols = ['Interval','Point of Reference']
# result_df.columns = df_cols
# result_df_grouped = [i[1].values.tolist() for i in result_df.groupby('Point of Reference')]
# result_df_grouped = [[i[0],i[-1]] for i in result_df_grouped]
# result_df_grouped = [i for sublst in result_df_grouped for i in sublst]
# # # return to str
# result_df = pd.DataFrame(result_df_grouped)
# result_df.columns = df_cols
# 
# x = result_df['Interval'].tolist()
# y = result_df['Point of Reference'].tolist()
# fig = px.line(
#             result_df, 
#             x="Interval", 
#             y="Point of Reference", 
#             color='Point of Reference', 
#             markers=True, 
#             #title='k: '+ str(k) + ' ' + str(attr_values) + ' interactions',
#             )
# fig.update_traces(textposition="bottom right", marker_size=15, line=dict(width=2.5), line_color="#4169E1")
# fig.update_layout(
#     xaxis = dict(
#         tickmode = 'array',
#         tickvals = [i for i in interval],
#         ticktext = [i.upper() for i in interval],
#         ticks = "outside", ticklen=10
#     ),
#     yaxis = dict(
#         tickmode = 'array',
#         tickvals = [i for i in interval],
#         ticktext = [i.upper() for i in interval],
#         ticks = "outside", ticklen=10
#     ),
# 
# )
# 
# #fig.update_xaxes(tickangle= -90)
# fig.update_layout(showlegend=False, font_size=20, plot_bgcolor='rgba(0,0,0,0)', autosize=False, width=750, height=600)
# fig.update_xaxes(showgrid=True, gridcolor='lightgrey', showline=True, linecolor = 'black', zeroline=False, mirror=True)
# fig.update_yaxes(showgrid=True, gridcolor='lightgrey', showline=True, linecolor = 'black', zeroline=False, mirror=True)
# fig.update_layout(xaxis_range=[1, 17],yaxis_range=[1, 17])
# fig.show()
# =============================================================================



# =============================================================================
# def Growth_UN_MAX(attr_val,stc,nodes,edges,time_inv):
#     c=0
#     intvls = []
#     for i in range(1,len(edges.columns)+1-c):
#         intvls.append([list(edges.columns[:i]), list(edges.columns[i:i+1])])
#         c += 1
#     intvls = intvls[:-1]
#     
#     skyline = {i:[] for i in range(1, len(edges.columns))}
#     dominate_counter = {}
#     for left,right in intvls:
#         max_length = len(left)
#         while len(left) >= 1:
#             #print(left)
#             diff,tia_diff = Diff_Static(nodes,edges,time_inv,right,left)
#             if not diff[1].empty:
#                 agg_diff = Aggregate_Static_Dist(diff,tia_diff,stc)
#                 if attr_val in agg_diff[1].index:
#                     current_w = agg_diff[1].loc[attr_val,:][0]
#                     dominate_counter[str((current_w,left,right))] = 0
#                     pr = len(left)
#                     #print('pr: ', pr)
#                     while not skyline[pr] and pr <= max_length:
#                         pr += 1
#                         #print('while..., pr: ', pr)
#                         if pr > max_length:
#                             break
#                     if pr > max_length:
#                         #print(pr, '>', max_length)
#                         previous_w = 0
#                     else:
#                         #print(pr, '<=', max_length)
#                         previous_w = skyline[pr][0][0]
#                     if current_w > previous_w:
#                         #print(current_w, '>', previous_w)
#                         if len(left) == pr:
#                             dominate_counter[str((current_w,left,right))] += 1
#                             dominate_counter[str((current_w,left,right))] += len(skyline[pr])
#                         skyline[len(left)] = [[current_w,left,right]]
#                     elif current_w == previous_w:
#                         if len(left) == pr:
#                             skyline.setdefault(len(left),[]).append([current_w,left,right])
#                     else:
#                         #print(current_w, '<=', previous_w)
#                         for s in skyline[pr]:
#                             dominate_counter[str(tuple(s))] += 1
#                     if len(left) > 1:
#                         pr2 = len(left)-1
#                         while not skyline[pr2] and pr2 >= 1:
#                             pr2 -= 1
#                             if pr2 == 0:
#                                 break
#                         if pr2 == 0:
#                             break
#                         else:
#                             if skyline[pr2][0][0] <= current_w:
#                                 dominate_counter[str((current_w,left,right))] += 1
#                                 dominate_counter[str((current_w,left,right))] += len(skyline[pr2])
#                                 skyline[pr2] = []
#             left = left[1:]
#     return(skyline,dominate_counter)
# =============================================================================


# =============================================================================
# def Shrink_UN_MIN(attr_val,stc,nodes,edges,time_inv):
#     s = [[str(i)] for i in edges.columns[:-1]]
#     e = [[str(i)] for i in edges.columns[1:]]
#     intvls = list(zip(s,e))
#     intvls = [list(i) for i in intvls]
#     
#     skyline = {i:[] for i in range(1, len(edges.columns))}
#     dominate_counter = {}
#     for left,right in intvls:
#         min_length = len(left)
#         flag = True
#         while flag == True:
#             #print('left',left,'right',right)
#             diff,tia_diff = Diff_Static(nodes,edges,time_inv,left,right)
#             if not diff[1].empty:
#                 #print('diff empty?', diff[1].empty)
#                 agg_diff = Aggregate_Static_Dist(diff,tia_diff,stc)
#                 if attr_val in agg_diff[1].index:
#                     current_w = agg_diff[1].loc[attr_val,:][0]
#                     dominate_counter[str((current_w,left,right))] = 0
#                     pr = len(left)
#                     #print('pr: ', pr)
#                     while not skyline[pr] and pr >= min_length:
#                         pr -= 1
#                         #print('while..., pr: ', pr)
#                         if pr < min_length:
#                             break
#                     if pr < min_length:
#                         #print(pr, '>', max_length)
#                         previous_w = 0
#                     else:
#                         #print(pr, '<=', max_length)
#                         previous_w = skyline[pr][0][0]
#                     if current_w > previous_w:
#                         #print(current_w, '>', previous_w)
#                         if len(left) == pr:
#                             dominate_counter[str((current_w,left,right))] += 1
#                             dominate_counter[str((current_w,left,right))] += len(skyline[pr])
#                         skyline[len(left)] = [[current_w,left,right]]
#                     elif current_w == previous_w:
#                         if len(left) == pr:
#                             skyline.setdefault(len(left),[]).append([current_w,left,right])
#                     else:
#                         #print(current_w, '<=', previous_w)
#                         for s in skyline[pr]:
#                             dominate_counter[str(tuple(s))] += 1
#                     # if left[-1] not reached the end --> left != right - 1
#                     if left[-1] != list(edges.columns)[list(edges.columns).index(right[0])-1]:
#                         pr2 = len(left)+1
#                         while not skyline[pr2] and pr2 <= len(list(
#                                 edges.columns)[list(edges.columns).index(left[0])
#                                                   :list(edges.columns).index(right[0])]):
#                             pr2 += 1
#                             if pr2 == len(list(
#                                     edges.columns)[list(edges.columns).index(left[0])
#                                                       :list(edges.columns).index(right[0])+1]):
#                                 break
#                         if pr2 == len(list(
#                                 edges.columns)[list(edges.columns).index(left[0])
#                                                   :list(edges.columns).index(right[0])+1]):
#                             break
#                         else:
#                             if skyline[pr2][0][0] <= current_w:
#                                 dominate_counter[str((current_w,left,right))] += 1
#                                 dominate_counter[str((current_w,left,right))] += len(skyline[pr2])
#                                 skyline[pr2] = []
#             if left[0] != edges.columns[0]:
#                 flag = True
#                 left = [edges.columns[list(edges.columns).index(left[0])-1]]+left
#             else:
#                 flag = False
#     return(skyline,dominate_counter)
# =============================================================================



# =============================================================================
# # shrinkage (told(union) - tnew) (minimal)
# def Shrink_UN_MIN(attr_val,stc,nodes,edges,time_inv):
#     s = [[str(i)] for i in edges.columns[:-1]]
#     e = [[str(i)] for i in edges.columns[1:]]
#     intvls = list(zip(s,e))
#     intvls = [list(i) for i in intvls]
#     
#     skyline = {i:[] for i in range(1, len(edges.columns))}
#     dominate_counter = {}
#     for left,right in intvls:
#         min_length = len(left)
#         flag = True
#         while flag == True:
#             #print('left',left,'right',right)
#             diff,tia_diff = Diff_Static(nodes,edges,time_inv,left,right)
#             if not diff[1].empty:
#                 #print('diff empty?', diff[1].empty)
#                 agg_diff = Aggregate_Static_Dist(diff,tia_diff,stc)
#                 if attr_val in agg_diff[1].index:
#                     current_w = agg_diff[1].loc[attr_val,:][0]
#                     dominate_counter[str((current_w,left,right))] = 0
#                     pr = len(left)
#                     #print('pr: ', pr)
#                     while not skyline[pr] and pr >= min_length:
#                         pr -= 1
#                         #print('while..., pr: ', pr)
#                         if pr < min_length:
#                             break
#                     if pr < min_length:
#                         #print(pr, '>', max_length)
#                         previous_w = 0
#                     else:
#                         #print(pr, '<=', max_length)
#                         previous_w = skyline[pr][0][0]
#                     if current_w > previous_w:
#                         #print(current_w, '>', previous_w)
#                         if len(left) == pr:
#                             dominate_counter[str((current_w,left,right))] += 1
#                             for s in skyline[pr]:
#                                 dominate_counter[str((current_w,left,right))] += dominate_counter[str(tuple(s))]
#                             dominate_counter[str(tuple(s))] = 0
#                         skyline[len(left)] = [[current_w,left,right]]
#                     elif current_w == previous_w:
#                         if len(left) == pr:
#                             skyline.setdefault(len(left),[]).append([current_w,left,right])
#                     else:
#                         #print(current_w, '<=', previous_w)
#                         for s in skyline[pr]:
#                             dominate_counter[str(tuple(s))] += 1
#                     # if left[-1] not reached the end
#                     if left[-1] != list(edges.columns)[list(edges.columns).index(right[0])-1]:
#                         pr2 = len(left)+1
#                         while not skyline[pr2] and pr2 <= len(list(
#                                 edges.columns)[list(edges.columns).index(left[0])
#                                                   :list(edges.columns).index(right[0])]):
#                             pr2 += 1
#                             if pr2 == len(list(
#                                     edges.columns)[list(edges.columns).index(left[0])
#                                                       :list(edges.columns).index(right[0])+1]):
#                                 break
#                         if pr2 == len(list(
#                                 edges.columns)[list(edges.columns).index(left[0])
#                                                   :list(edges.columns).index(right[0])+1]):
#                             break
#                         else:
#                             if skyline[pr2][0][0] <= current_w:
#                                 dominate_counter[str((current_w,left,right))] += 1
#                                 for s in skyline[pr2]:
#                                     dominate_counter[str((current_w,left,right))] += dominate_counter[str(tuple(s))]
#                                 dominate_counter[str(tuple(s))] = 0
#                                 skyline[pr2] = []
#             if left[0] != edges.columns[0]:
#                 flag = True
#                 left = [edges.columns[list(edges.columns).index(left[0])-1]]+left
#             else:
#                 flag = False
#     skyline = {i:j for i,j in skyline.items() if j}
#     dominate_counter = {i:j for i,j in dominate_counter.items() \
#                         if list(eval(i)) in [si for s in skyline.values() for si in s]}
#     return(skyline,dominate_counter)
# =============================================================================
