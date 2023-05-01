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
import gc


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

# primary school dataset
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
x = [('1A','1A'), ('1B','1B'), ('2A','2A'), ('2B','2B'), ('3A','3A'), \
     ('3B','3B'), ('4A','4A'), ('4B','4B'), ('5A','5A'), ('5B','5B')]


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

skyline_stab, dominance_stab = Stab_INX_MAX(attr_values,stc_attrs,nodes_df,edges_df,time_invariant_attr)


# Stability (told(un)&tnew) (minimal)

attr_values = ('F', 'F')
stc_attrs = ['gender']


#def Stab_UN_MIN(attr_val,stc,nodes,edges,time_inv):
    
def Intersection_Static_UN(nodesdf,edgesdf,tia,intvl):
    nodes_u = nodesdf[intvl[0]][nodesdf[intvl[0]].any(axis=1)]
    n = pd.merge(nodes_u, nodesdf.loc[:,intvl[1]], left_index=True, right_index=True)
    e = edgesdf[intvl[0]+intvl[1]][edgesdf[intvl[0]].any(axis=1)]
    e = e[e.loc[:,intvl[1]]==1]
    tiainx = tia[tia.index.isin(n.index)]
    ne = [n,e]
    return(ne,tiainx)

def Stab_INX_U_MIN(attr_val,stc,nodes,edges,tia):
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
            inx,tia_inx = Intersection_Static_UN(nodes,edges,tia,[left,right])
            if not inx[1].empty:
                agg_inx = Aggregate_Static_Dist(inx,tia_inx,stc)
                if attr_val in agg_inx[1].index:
                    current_w = agg_inx[1].loc[attr_val,:][0]
                    dominate_counter[str((current_w,left,right))] = 0
                    pr = len(left)
                    while not skyline[pr] and pr >= min_length:
                        pr -= 1
                        if pr < min_length:
                            break
                    if pr < min_length:
                        previous_w = 0
                    else:
                        previous_w = skyline[pr][0][0]
                    if current_w > previous_w:
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
                        for s in skyline[pr]:
                            dominate_counter[str(tuple(s))] += 1
                    if left[0] != edges.columns[0]:
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

skyline_stabU, dominance_stabU = Stab_INX_U_MIN(attr_values,stc_attrs,nodes_df,edges_df,time_invariant_attr)
                

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
            diff,tia_diff = Diff_Static(nodes,edges,time_inv,right,left)
            if not diff[1].empty:
                agg_diff = Aggregate_Static_Dist(diff,tia_diff,stc)
                if attr_val in agg_diff[1].index:
                    current_w = agg_diff[1].loc[attr_val,:][0]
                    dominate_counter[str((current_w,left,right))] = 0
                    pr = len(left)
                    while not skyline[pr] and pr <= max_length:
                        pr += 1
                        if pr > max_length:
                            break
                    if pr > max_length:
                        previous_w = 0
                    else:
                        previous_w = skyline[pr][0][0]
                    if current_w > previous_w:
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
            diff,tia_diff = Diff_Static(nodes,edges,time_inv,left,right)
            if not diff[1].empty:
                agg_diff = Aggregate_Static_Dist(diff,tia_diff,stc)
                if attr_val in agg_diff[1].index:
                    current_w = agg_diff[1].loc[attr_val,:][0]
                    dominate_counter[str((current_w,left,right))] = 0
                    pr = len(left)
                    while not skyline[pr] and pr >= min_length:
                        pr -= 1
                        if pr < min_length:
                            break
                    if pr < min_length:
                        previous_w = 0
                    else:
                        previous_w = skyline[pr][0][0]
                    if current_w > previous_w:
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
                        for s in skyline[pr]:
                            dominate_counter[str(tuple(s))] += 1
                    if left[0] != edges.columns[0]:
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

# top-k

k = 3
dom_val = sorted([i[1] for i in dominate_counter.items()])[::-1][:k]

top={}
for k,val in dominate_counter.items():
    if val in dom_val:
        top[k] = val
    

############################


# 1 TOP-K DOMINATING

x = [('F', 'F'), ('F', 'M'), ('M', 'F'), ('M', 'M')]
stc_attrs = ['gender']
dataset = 'movielens'

x = [('1A','1A'), ('1B','1B'), ('2A','2A'), ('2B','2B'), ('3A','3A'), \
     ('3B','3B'), ('4A','4A'), ('4B','4B'), ('5A','5A'), ('5B','5B')]
stc_attrs = ['class']
dataset = 'primary'

dataset = 'dblp'

top_stab = {}
for xi in x:
    skyline_stab, dominance_stab = Stab_INX_MAX(xi,stc_attrs,nodes_df,edges_df,time_invariant_attr)
    size = sum([len(i) for i in skyline_stab.values()])
    top_stab[str(xi)] = [skyline_stab, dominance_stab, size]
    print('size stab', xi, size)

df = pd.DataFrame(top_stab).T
# if file does not exist write header 
if not os.path.isfile('experiments/qualitative/'+dataset+'/stab_'+stc_attrs[0]+'_sky.csv'):
   df.to_csv('experiments/qualitative/'+dataset+'/stab_'+stc_attrs[0]+'_sky.csv', header='column_names')
else: # else it exists so append without writing the header
   df.to_csv('experiments/qualitative/'+dataset+'/stab_'+stc_attrs[0]+'_sky.csv', mode='a', header='column_names')

top_grow = {}
for xi in x:
    skyline_grow, dominance_grow = Growth_UN_MAX(xi,stc_attrs,nodes_df,edges_df,time_invariant_attr)
    size = sum([len(i) for i in skyline_grow.values()])
    top_grow[str(xi)] = [skyline_grow, dominance_grow, size]
    print('size grow', xi, size)

df = pd.DataFrame(top_grow).T
# if file does not exist write header 
if not os.path.isfile('experiments/qualitative/'+dataset+'/grow_'+stc_attrs[0]+'_sky.csv'):
   df.to_csv('experiments/qualitative/'+dataset+'/grow_'+stc_attrs[0]+'_sky.csv', header='column_names')
else: # else it exists so append without writing the header
   df.to_csv('experiments/qualitative/'+dataset+'/grow_'+stc_attrs[0]+'_sky.csv', mode='a', header='column_names')

top_shr = {}
for xi in x:
    skyline_shr, dominance_shr = Shrink_UN_MIN(xi,stc_attrs,nodes_df,edges_df,time_invariant_attr)
    size = sum([len(i) for i in skyline_shr.values()])
    top_shr[str(xi)] = [skyline_shr, dominance_shr, size]
    print('size shr', xi, size)

df = pd.DataFrame(top_shr).T
# if file does not exist write header 
if not os.path.isfile('experiments/qualitative/'+dataset+'/shr_'+stc_attrs[0]+'_sky.csv'):
   df.to_csv('experiments/qualitative/'+dataset+'/shr_'+stc_attrs[0]+'_sky.csv', header='column_names')
else: # else it exists so append without writing the header
   df.to_csv('experiments/qualitative/'+dataset+'/shr_'+stc_attrs[0]+'_sky.csv', mode='a', header='column_names')


top_stabU = {}
for xi in x:
    skyline_stabU, dominance_stabU = Stab_INX_U_MIN(xi,stc_attrs,nodes_df,edges_df,time_invariant_attr)
    size = sum([len(i) for i in skyline_stabU.values()])
    top_stabU[str(xi)] = [skyline_stabU, dominance_stabU, size]
    print('size stabU', xi, size)

df = pd.DataFrame(top_stabU).T
# if file does not exist write header 
if not os.path.isfile('experiments/qualitative/'+dataset+'/stabU_'+stc_attrs[0]+'_sky.csv'):
   df.to_csv('experiments/qualitative/'+dataset+'/stabU_'+stc_attrs[0]+'_sky.csv', header='column_names')
else: # else it exists so append without writing the header
   df.to_csv('experiments/qualitative/'+dataset+'/stabU_'+stc_attrs[0]+'_sky.csv', mode='a', header='column_names')




# 2 TIME & SIZE EXPERIMENTS FOR DBLP | RUNTIME

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
            gc.disable()
            start = time.perf_counter_ns()
            skyline_stab, dominance_stab = Stab_INX_MAX(xi,stc_attrs,nodes_df,y,time_invariant_attr)
            end = time.perf_counter_ns()
            gc.enable()
            start_end_agg.append(end-start)
        result.append(start_end_agg)
    
    res = pd.DataFrame(result).T
    res = res*(1e-9)
    res[str(xi)+'_avg'] = res.mean(axis=1)
    res[str(xi)+'_min'] = res.min(axis=1)
    res = res.round(2)

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
            gc.disable()
            start = time.perf_counter_ns()
            skyline_grow, dominance_grow = Growth_UN_MAX(xi,stc_attrs,nodes_df,y,time_invariant_attr)
            end = time.perf_counter_ns()
            gc.enable()
            start_end_agg.append(end-start)
        result.append(start_end_agg)
    
    res = pd.DataFrame(result).T
    res = res*(1e-9)
    res[str(xi)+'_avg'] = res.mean(axis=1)
    res[str(xi)+'_min'] = res.min(axis=1)
    res = res.round(2)

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
            gc.disable()
            start = time.perf_counter_ns()
            skyline_shr, dominance_shr = Shrink_UN_MIN(xi,stc_attrs,nodes_df,y,time_invariant_attr)
            end = time.perf_counter_ns()
            gc.enable()
            start_end_agg.append(end-start)
        result.append(start_end_agg)
    
    res = pd.DataFrame(result).T
    res = res*(1e-9)
    res[str(xi)+'_avg'] = res.mean(axis=1)
    res[str(xi)+'_min'] = res.min(axis=1)
    res = res.round(2)

    # if file does not exist write header 
    if not os.path.isfile('experiments/runtime/dblp/shr_gender.csv'):
       res.to_csv('experiments/runtime/dblp/shr_gender.csv', header='column_names')
    else: # else it exists so append without writing the header
       res.to_csv('experiments/runtime/dblp/shr_gender.csv', mode='a', header='column_names')

# StabU

for xi in x:
    result = []
    for j in range(5):
        start_end_agg = []
        for y in edges_sliced:
            gc.disable()
            start = time.perf_counter_ns()
            skyline_stabU, dominance_stabU = Stab_INX_U_MIN(xi,stc_attrs,nodes_df,y,time_invariant_attr)
            end = time.perf_counter_ns()
            gc.enable()
            start_end_agg.append(end-start)
        result.append(start_end_agg)

    res = pd.DataFrame(result).T
    res = res*(1e-9)
    res[str(xi)+'_avg'] = res.mean(axis=1)
    res[str(xi)+'_min'] = res.min(axis=1)
    res = res.round(2)

    # if file does not exist write header 
    if not os.path.isfile('experiments/runtime/dblp/stabU_gender.csv'):
       res.to_csv('experiments/runtime/dblp/stabU_gender.csv', header='column_names')
    else: # else it exists so append without writing the header
       res.to_csv('experiments/runtime/dblp/stabU_gender.csv', mode='a', header='column_names')



# Primary School & MovieLens RUNTIME for full dataset

x = [('F', 'F'), ('F', 'M'), ('M', 'F'), ('M', 'M')]
attribute = 'gender'
stc_attrs = ['gender']

x = [('1A','1A'), ('1B','1B'), ('2A','2A'), ('2B','2B'), ('3A','3A'), \
     ('3B','3B'), ('4A','4A'), ('4B','4B'), ('5A','5A'), ('5B','5B')]
attribute = 'class'
stc_attrs = ['class']

dataset = 'primary'
dataset = 'movielens'

for xi in x:
    result = []
    for j in range(5):
        start_end_agg = []
        gc.disable()
        start = time.perf_counter_ns()
        skyline_stab, dominance_stab = Stab_INX_MAX(xi,stc_attrs,nodes_df,edges_df,time_invariant_attr)
        end = time.perf_counter_ns()
        gc.enable()
        start_end_agg.append(end-start)
        result.append(start_end_agg)
    
    res = pd.DataFrame(result).T
    res = res*(1e-9)
    res[str(xi)+'_avg'] = res.mean(axis=1)
    #res[str(xi)+'_min'] = res.min(axis=1)
    res = res.round(2)

    # if file does not exist write header 
    if not os.path.isfile('experiments/runtime/'+dataset+'/stab_'+attribute+'.csv'):
       res.to_csv('experiments/runtime/'+dataset+'/stab_'+attribute+'.csv', header='column_names')
    else: # else it exists so append without writing the header
       res.to_csv('experiments/runtime/'+dataset+'/stab_'+attribute+'.csv', mode='a', header='column_names')


for xi in x:
    result = []
    for j in range(5):
        start_end_agg = []
        gc.disable()
        start = time.perf_counter_ns()
        skyline_grow, dominance_grow = Growth_UN_MAX(xi,stc_attrs,nodes_df,edges_df,time_invariant_attr)
        end = time.perf_counter_ns()
        gc.enable()
        start_end_agg.append(end-start)
        result.append(start_end_agg)
    
    res = pd.DataFrame(result).T
    res = res*(1e-9)
    res[str(xi)+'_avg'] = res.mean(axis=1)
    #res[str(xi)+'_min'] = res.min(axis=1)
    res = res.round(2)

    # if file does not exist write header 
    if not os.path.isfile('experiments/runtime/'+dataset+'/grow_'+attribute+'.csv'):
       res.to_csv('experiments/runtime/'+dataset+'/grow_'+attribute+'.csv', header='column_names')
    else: # else it exists so append without writing the header
       res.to_csv('experiments/runtime/'+dataset+'/grow_'+attribute+'.csv', mode='a', header='column_names')


for xi in x:
    result = []
    for j in range(5):
        start_end_agg = []
        gc.disable()
        start = time.perf_counter_ns()
        skyline_shr, dominance_shr = Shrink_UN_MIN(xi,stc_attrs,nodes_df,edges_df,time_invariant_attr)
        end = time.perf_counter_ns()
        gc.enable()
        start_end_agg.append(end-start)
        result.append(start_end_agg)
    
    res = pd.DataFrame(result).T
    res = res*(1e-9)
    res[str(xi)+'_avg'] = res.mean(axis=1)
    #res[str(xi)+'_min'] = res.min(axis=1)
    res = res.round(2)

    # if file does not exist write header 
    if not os.path.isfile('experiments/runtime/'+dataset+'/shr_'+attribute+'.csv'):
       res.to_csv('experiments/runtime/'+dataset+'/shr_'+attribute+'.csv', header='column_names')
    else: # else it exists so append without writing the header
       res.to_csv('experiments/runtime/'+dataset+'/shr_'+attribute+'.csv', mode='a', header='column_names')


# StabU

for xi in x:
    result = []
    for j in range(5):
        start_end_agg = []
        gc.disable()
        start = time.perf_counter_ns()
        skyline_stabU, dominance_stabU = Stab_INX_U_MIN(xi,stc_attrs,nodes_df,edges_df,time_invariant_attr)
        end = time.perf_counter_ns()
        gc.enable()
        start_end_agg.append(end-start)
        result.append(start_end_agg)

    res = pd.DataFrame(result).T
    res = res*(1e-9)
    res[str(xi)+'_avg'] = res.mean(axis=1)
    #res[str(xi)+'_min'] = res.min(axis=1)
    res = res.round(2)

    # if file does not exist write header 
    if not os.path.isfile('experiments/runtime/'+dataset+'/stabU_'+attribute+'.csv'):
       res.to_csv('experiments/runtime/'+dataset+'/stabU_'+attribute+'.csv', header='column_names')
    else: # else it exists so append without writing the header
       res.to_csv('experiments/runtime/'+dataset+'/stabU_'+attribute+'.csv', mode='a', header='column_names')



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


# ONE-PASS

# STABILITY

def Stab_one_pass(attr_val_combs,stc_attrs,nodes_df,edges_df,time_invariant_attr):
    c=0
    intvls = []
    for i in range(1,len(edges_df.columns)+1-c):
        intvls.append([list(edges_df.columns[:i]), list(edges_df.columns[i:i+1])])
        c += 1
    intvls = intvls[:-1]
    
    skyline = {j:{i:[] for i in range(1, len(edges_df.columns))} for j in attr_val_combs}
    dominate_counter = {i:{} for i in attr_val_combs}
    for left,right in intvls:
        max_length = len(left)
        while len(left) >= 1:
            current_w = {}
            previous_w = {}
            #print(left)
            inx,tia_inx = Intersection_Static(nodes_df,edges_df,time_invariant_attr,left+right)
            if not inx[1].empty:
                agg_inx = Aggregate_Static_Dist(inx,tia_inx,stc_attrs)
                for comb in attr_val_combs:
                    if comb in agg_inx[1].index:
                        current_w[comb] = agg_inx[1].loc[comb,:][0]
                        dominate_counter[comb][str((current_w[comb],left,right))] = 0
                        pr = len(left)
                        while not skyline[comb][pr] and pr <= max_length:
                            pr += 1
                            #print('while..., pr: ', pr)
                            if pr > max_length:
                                break
                        if pr > max_length:
                            previous_w[comb] = 0
                        else:
                            previous_w[comb] = skyline[comb][pr][0][0]
                        if current_w[comb] > previous_w[comb]:
                            if len(left) == pr:
                                dominate_counter[comb][str((current_w[comb],left,right))] += 1
                                for s in skyline[comb][pr]:
                                    dominate_counter[comb][str((current_w[comb],left,right))] += dominate_counter[comb][str(tuple(s))]
                                #del dominate_counter[str(tuple(s))]
                                dominate_counter[comb][str(tuple(s))] = 0
                            skyline[comb][len(left)] = [[current_w[comb],left,right]]
                        elif current_w[comb] == previous_w[comb]:
                            if len(left) == pr:
                                skyline[comb].setdefault(len(left),[]).append([current_w[comb],left,right])
                        else:
                            for s in skyline[comb][pr]:
                                dominate_counter[comb][str(tuple(s))] += 1
                        if len(left) > 1:
                            pr2 = len(left)-1
                            while not skyline[comb][pr2] and pr2 >= 1:
                                pr2 -= 1
                                if pr2 == 0:
                                    break
                            if pr2 > 0:
                                if skyline[comb][pr2][0][0] <= current_w[comb]:
                                    dominate_counter[comb][str((current_w[comb],left,right))] += 1
                                    for s in skyline[comb][pr2]:
                                        dominate_counter[comb][str((current_w[comb],left,right))] += dominate_counter[comb][str(tuple(s))]
                                    #del dominate_counter[str(tuple(s))]
                                    dominate_counter[comb][str(tuple(s))] = 0
                                    skyline[comb][pr2] = []            
            left = left[1:]
    for comb in attr_val_combs:
        skyline[comb] = {i:j for i,j in skyline[comb].items() if j}
        dominate_counter[comb] = {i:j for i,j in dominate_counter[comb].items() \
                        if list(eval(i)) in [si for s in skyline[comb].values() for si in s]}
    return(skyline,dominate_counter)


# GROWTH

def Grow_one_pass(attr_val_combs,stc_attrs,nodes_df,edges_df,time_invariant_attr):
    c=0
    intvls = []
    for i in range(1,len(edges_df.columns)+1-c):
        intvls.append([list(edges_df.columns[:i]), list(edges_df.columns[i:i+1])])
        c += 1
    intvls = intvls[:-1]
    
    skyline = {j:{i:[] for i in range(1, len(edges_df.columns))} for j in attr_val_combs}
    dominate_counter = {i:{} for i in attr_val_combs}
    for left,right in intvls:
        max_length = len(left)
        while len(left) >= 1:
            current_w = {}
            previous_w = {}
            diff,tia_diff = Diff_Static(nodes_df,edges_df,time_invariant_attr,right,left)
            if not diff[1].empty:
                agg_diff = Aggregate_Static_Dist(diff,tia_diff,stc_attrs)
                for comb in attr_val_combs:
                    if comb in agg_diff[1].index:
                        current_w[comb] = agg_diff[1].loc[comb,:][0]
                        dominate_counter[comb][str((current_w[comb],left,right))] = 0
                        pr = len(left)
                        while not skyline[comb][pr] and pr <= max_length:
                            pr += 1
                            if pr > max_length:
                                break
                        if pr > max_length:
                            previous_w[comb] = 0
                        else:
                            previous_w[comb] = skyline[comb][pr][0][0]
                        if current_w[comb] > previous_w[comb]:
                            if len(left) == pr:
                                dominate_counter[comb][str((current_w[comb],left,right))] += 1
                                for s in skyline[comb][pr]:
                                    dominate_counter[comb][str((current_w[comb],left,right))] += dominate_counter[comb][str(tuple(s))]
                                #del dominate_counter[str(tuple(s))]
                                dominate_counter[comb][str(tuple(s))] = 0
                            skyline[comb][len(left)] = [[current_w[comb],left,right]]
                        elif current_w[comb] == previous_w[comb]:
                            if len(left) == pr:
                                skyline[comb].setdefault(len(left),[]).append([current_w[comb],left,right])
                        else:
                            for s in skyline[comb][pr]:
                                dominate_counter[comb][str(tuple(s))] += 1
                        if len(left) > 1:
                            pr2 = len(left)-1
                            while not skyline[comb][pr2] and pr2 >= 1:
                                pr2 -= 1
                                if pr2 == 0:
                                    break
                            if pr2 > 0:
                                if skyline[comb][pr2][0][0] <= current_w[comb]:
                                    dominate_counter[comb][str((current_w[comb],left,right))] += 1
                                    for s in skyline[comb][pr2]:
                                        dominate_counter[comb][str((current_w[comb],left,right))] += dominate_counter[comb][str(tuple(s))]
                                    #del dominate_counter[str(tuple(s))]
                                    dominate_counter[comb][str(tuple(s))] = 0
                                    skyline[comb][pr2] = []            
            left = left[1:]
    for comb in attr_val_combs:
        skyline[comb] = {i:j for i,j in skyline[comb].items() if j}
        dominate_counter[comb] = {i:j for i,j in dominate_counter[comb].items() \
                        if list(eval(i)) in [si for s in skyline[comb].values() for si in s]}
    return(skyline,dominate_counter)


# SHRINKAGE

def Shr_one_pass(attr_val_combs,stc_attrs,nodes_df,edges_df,time_invariant_attr):
    s = [[str(i)] for i in edges_df.columns[:-1]]
    e = [[str(i)] for i in edges_df.columns[1:]]
    intvls = list(zip(s,e))
    intvls = [list(i) for i in intvls]
    
    skyline = {j:{i:[] for i in range(1, len(edges_df.columns))} for j in attr_val_combs}
    dominate_counter = {i:{} for i in attr_val_combs}
    
    for left,right in intvls:
        min_length = len(left)
        flag = True
        while flag == True:
            current_w = {}
            previous_w = {}
            diff,tia_diff = Diff_Static(nodes_df,edges_df,time_invariant_attr,left,right)
            if not diff[1].empty:
                agg_diff = Aggregate_Static_Dist(diff,tia_diff,stc_attrs)
                for comb in attr_val_combs:
                    if comb in agg_diff[1].index:
                        current_w[comb]= agg_diff[1].loc[comb,:][0]
                        dominate_counter[comb][str((current_w[comb],left,right))] = 0
                        pr = len(left)
                        while not skyline[comb][pr] and pr >= min_length:
                            pr -= 1
                            if pr < min_length:
                                break
                        if pr < min_length:
                            previous_w[comb] = 0
                        else:
                            previous_w[comb] = skyline[comb][pr][0][0]
                        if current_w[comb] > previous_w[comb]:
                            if len(left) == pr:
                                dominate_counter[comb][str((current_w[comb],left,right))] += 1
                                for s in skyline[comb][pr]:
                                    dominate_counter[comb][str((current_w[comb],left,right))] += dominate_counter[comb][str(tuple(s))]
                                dominate_counter[comb][str(tuple(s))] = 0
                            skyline[comb][len(left)] = [[current_w[comb],left,right]]
                        elif current_w[comb] == previous_w[comb]:
                            if len(left) == pr:
                                skyline[comb].setdefault(len(left),[]).append([current_w[comb],left,right])
                        else:
                            for s in skyline[comb][pr]:
                                dominate_counter[comb][str(tuple(s))] += 1
                        if left[0] != edges_df.columns[0]:
                            pr2 = len(left)+1
                            while not skyline[comb][pr2] and pr2 <= len(list(edges_df.columns)[:list(edges_df.columns).index(right[0])]):
                                pr2 += 1
                                if pr2 == len(list(edges_df.columns)[:list(edges_df.columns).index(right[0])])+1:
                                    break
                            if pr2 < len(list(edges_df.columns)[:list(edges_df.columns).index(right[0])])+1:
                                if skyline[comb][pr2][0][0] <= current_w[comb]:
                                    dominate_counter[comb][str((current_w[comb],left,right))] += 1
                                    for s in skyline[comb][pr2]:
                                        dominate_counter[comb][str((current_w[comb],left,right))] += dominate_counter[comb][str(tuple(s))]
                                    dominate_counter[comb][str(tuple(s))] = 0
                                    skyline[comb][pr2] = []
            if left[0] != edges_df.columns[0]:
                flag = True
                left = [edges_df.columns[list(edges_df.columns).index(left[0])-1]]+left
            else:
                flag = False
    for comb in attr_val_combs:
        skyline[comb] = {i:j for i,j in skyline[comb].items() if j}
        dominate_counter[comb] = {i:j for i,j in dominate_counter[comb].items() \
                            if list(eval(i)) in [si for s in skyline[comb].values() for si in s]}
    return(skyline,dominate_counter)




# RUNTIME ONE-PASS

stc_attrs = ['gender']
attr_val = list(time_invariant_attr[stc_attrs].value_counts().index)
attr_val = [('M'), ('F')]
attr_val_combs = list(itertools.product(attr_val, repeat=2))
attr_val_combs = [tuple([i[0][0],i[1][0]]) for i in attr_val_combs]

# DBLP

edges_sliced = [edges_df.iloc[:,:6], edges_df.iloc[:,:11], edges_df.iloc[:,:16], edges_df.iloc[:,:21]]
edges_sliced = [i.loc[~(i==0).all(axis=1)] for i in edges_sliced]

result = []
for j in range(5):
    start_end_agg = []
    for y in edges_sliced:
        gc.disable()
        start = time.perf_counter_ns()
        sky_onepass_stab, dom_onepass_stab = Stab_one_pass(attr_val_combs,stc_attrs,nodes_df,y,time_invariant_attr)
        end = time.perf_counter_ns()
        gc.enable()
        start_end_agg.append(end-start)
    result.append(start_end_agg)

res = pd.DataFrame(result).T
res = res*(1e-9)
res['all_avg'] = res.mean(axis=1)
res['all_min'] = res.min(axis=1)
res = res.round(2)

# if file does not exist write header 
if not os.path.isfile('experiments/runtime/one_pass/dblp/stab_gender.csv'):
   res.to_csv('experiments/runtime/one_pass/dblp/stab_gender.csv', header='column_names')
else: # else it exists so append without writing the header
   res.to_csv('experiments/runtime/one_pass/dblp/stab_gender.csv', mode='a', header='column_names')


result = []
for j in range(5):
    start_end_agg = []
    for y in edges_sliced:
        gc.disable()
        start = time.perf_counter_ns()
        sky_onepass_grow, dom_onepass_grow = Grow_one_pass(attr_val_combs,stc_attrs,nodes_df,y,time_invariant_attr)
        end = time.perf_counter_ns()
        gc.enable()
        start_end_agg.append(end-start)
    result.append(start_end_agg)

res = pd.DataFrame(result).T
res = res*(1e-9)
res['all_avg'] = res.mean(axis=1)
res['all_min'] = res.min(axis=1)
res = res.round(2)

# if file does not exist write header 
if not os.path.isfile('experiments/runtime/one_pass/dblp/grow_gender.csv'):
   res.to_csv('experiments/runtime/one_pass/dblp/grow_gender.csv', header='column_names')
else: # else it exists so append without writing the header
   res.to_csv('experiments/runtime/one_pass/dblp/grow_gender.csv', mode='a', header='column_names')


result = []
for j in range(5):
    start_end_agg = []
    for y in edges_sliced:
        gc.disable()
        start = time.perf_counter_ns()
        sky_onepass_shr, dom_onepass_shr = Shr_one_pass(attr_val_combs,stc_attrs,nodes_df,y,time_invariant_attr)
        end = time.perf_counter_ns()
        gc.enable()
        start_end_agg.append(end-start)
    result.append(start_end_agg)

res = pd.DataFrame(result).T
res = res*(1e-9)
res['all_avg'] = res.mean(axis=1)
res['all_min'] = res.min(axis=1)
res = res.round(2)

# if file does not exist write header 
if not os.path.isfile('experiments/one_pass/runtime/dblp/shr_gender.csv'):
   res.to_csv('experiments/runtime/one_pass/dblp/shr_gender.csv', header='column_names')
else: # else it exists so append without writing the header
   res.to_csv('experiments/runtime/one_pass/dblp/shr_gender.csv', mode='a', header='column_names')



# PRIMARY SCHOOL & MOVIELENS

dataset = 'primary'
dataset = 'movielens'

result = []
for j in range(5):
    start_end_agg = []
    gc.disable()
    start = time.perf_counter_ns()
    sky_onepass_stab, dom_onepass_stab = Stab_one_pass(attr_val_combs,stc_attrs,nodes_df,edges_df,time_invariant_attr)
    end = time.perf_counter_ns()
    gc.enable()
    start_end_agg.append(end-start)
    result.append(start_end_agg)

res = pd.DataFrame(result).T
res = res*(1e-9)
res['all_avg'] = res.mean(axis=1)
res['all_min'] = res.min(axis=1)
res = res.round(2)

# if file does not exist write header 
if not os.path.isfile('experiments/runtime/one_pass/'+dataset+'/stab_gender.csv'):
   res.to_csv('experiments/runtime/one_pass/'+dataset+'/stab_gender.csv', header='column_names')
else: # else it exists so append without writing the header
   res.to_csv('experiments/runtime/one_pass/'+dataset+'/stab_gender.csv', mode='a', header='column_names')


result = []
for j in range(5):
    start_end_agg = []
    gc.disable()
    start = time.perf_counter_ns()
    sky_onepass_grow, dom_onepass_grow = Grow_one_pass(attr_val_combs,stc_attrs,nodes_df,edges_df,time_invariant_attr)
    end = time.perf_counter_ns()
    gc.enable()
    start_end_agg.append(end-start)
    result.append(start_end_agg)

res = pd.DataFrame(result).T
res = res*(1e-9)
res['all_avg'] = res.mean(axis=1)
res['all_min'] = res.min(axis=1)
res = res.round(2)

# if file does not exist write header 
if not os.path.isfile('experiments/runtime/one_pass/'+dataset+'/grow_gender.csv'):
   res.to_csv('experiments/runtime/one_pass/'+dataset+'/grow_gender.csv', header='column_names')
else: # else it exists so append without writing the header
   res.to_csv('experiments/runtime/one_pass/'+dataset+'/grow_gender.csv', mode='a', header='column_names')


result = []
for j in range(5):
    start_end_agg = []
    gc.disable()
    start = time.perf_counter_ns()
    sky_onepass_shr, dom_onepass_shr = Shr_one_pass(attr_val_combs,stc_attrs,nodes_df,edges_df,time_invariant_attr)
    end = time.perf_counter_ns()
    gc.enable()
    start_end_agg.append(end-start)
    result.append(start_end_agg)

res = pd.DataFrame(result).T
res = res*(1e-9)
res['all_avg'] = res.mean(axis=1)
res['all_min'] = res.min(axis=1)
res = res.round(2)

# if file does not exist write header 
if not os.path.isfile('experiments/runtime/one_pass/'+dataset+'/shr_gender.csv'):
   res.to_csv('experiments/runtime/one_pass/'+dataset+'/shr_gender.csv', header='column_names')
else: # else it exists so append without writing the header
   res.to_csv('experiments/runtime/one_pass/'+dataset+'/shr_gender.csv', mode='a', header='column_names')