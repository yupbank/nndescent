import numpy as np
import pandas as pd


def euclidean(vec_a, vec_b):
    return np.linalg.norm(np.array(vec_a)-np.array(vec_b))


def calculate(df, data, dist=euclidean, dist_args=(), score_col='score'):
    df['src_vec'] = data[df['src']].tolist()
    df['dst_vec'] = data[df['dst']].tolist()
    df[score_col] = df.apply(lambda r: dist(
        r['src_vec'], r['dst_vec'], *dist_args), axis=1)
    df.drop(['src_vec', 'dst_vec'], axis=1, inplace=True)
    return df


def initial_graph(n_vertices, n_neighbors=10):
    start = np.repeat(np.arange(n_vertices), n_neighbors)
    end = np.array([np.random.choice(n_vertices, n_neighbors, replace=False)
                    for i in range(n_vertices)]).ravel()
    edges = np.vstack([start, end]).T
    return edges


def initial_edges(
        n_vertices,
        n_neighbors=10,
):
    n_vertices = data.shape[0]
    inital_edges = initial_graph(n_vertices, n_neighbors)
    df = pd.DataFrame(data=inital_edges, columns=['src', 'dst'])
    df.loc[:, 'is_new'] = True
    return df[['src', 'dst', 'is_new']]


def combine_pairs(row):
    new = row[row['is_new']]
    old = row[~row['is_new']]
    new_ids = new['dst'].to_numpy()
    old_ids = old['dst'].to_numpy()
    new_index = np.triu_indices(new_ids.shape[0], 1)
    src_new = new_ids[new_index[0]]
    dst_new = new_ids[new_index[1]]
    src_old, dst_old = np.meshgrid(new_ids, old_ids)
    srcs = np.hstack([src_new, src_old.ravel()])
    dsts = np.hstack([dst_new, dst_old.ravel()])
    return pd.DataFrame({'src': srcs, 'dst': dsts})


def add_random_column(df, col_name='rand'):
    df.loc[:, col_name] = np.random.randn(df.shape[0])
    return df


def merge_df(old_df, new_df, n_neighbors):
    old_df['is_listed'] = True
    new_df['is_listed'] = False
    total_df = pd.concat([old_df, new_df], sort=False)
    total_df['row_number'] = total_df.groupby('src')['score'].rank('first')
    df = total_df[total_df['row_number'] <= n_neighbors]
    c = df.agg({'is_listed': 'sum'})[0]
    print c
    return df, c


def process_one_iteration(df, size=2):
    df = df[['src', 'dst', 'is_new']]
    rand_col = 'rand'
    filter_row_col = 'row_number'
    df = add_random_column(df, rand_col)
    df[filter_row_col] = df.groupby(['src', 'is_new'])[
        rand_col].rank('first')
    # only remove tail edges in new group
    keep_condition = ~(df['is_new'] & (df[filter_row_col] > size))
    sampled = df[keep_condition]
    df.loc[keep_condition, 'is_new'] = False

    sampled = sampled[['src', 'dst', 'is_new']]

    sampled = add_random_column(sampled, rand_col)
    sampled[filter_row_col] = sampled.groupby(['dst', 'is_new'])[
        rand_col].rank('first')
    reverse_sampled = sampled[sampled[filter_row_col] <= size].rename(
        columns={'src': 'dst', 'dst': 'src'})[['src', 'dst', 'is_new']]

    overall_data = pd.concat([sampled, reverse_sampled], sort=False)
    new_stuff = overall_data.rename(
        columns={'src': 'local'}).groupby('local').apply(combine_pairs).reset_index()[['src', 'dst']]
    new_stuff = new_stuff.drop_duplicates()
    if new_stuff.shape[0] == 0:
        return df

    return new_stuff


def main(data, n_neighbors=10, rho=0.4, iter_num=2, dist=euclidean, dist_args=(),
         delta=0.001):
    size = int(n_neighbors*rho)
    n_vertices = data.shape[0]
    break_count = n_vertices*n_neighbors*delta
    print 'break_count', break_count
    df = initial_edges(data)
    df = calculate(df, data, dist, dist_args)
    for i in range(iter_num):
        new_stuff = process_one_iteration(df, size)
        new_df = pd.merge(
            new_stuff, df[['src', 'dst']],  how='outer', indicator=True)
        new_df = new_df.loc[new_df._merge == 'left_only', ['src', 'dst']]

        new_df = calculate(new_df, data, dist)
        new_df['is_new'] = True
        df, c = merge_df(df, new_df, n_neighbors)
        df = df[
            ['src', 'dst', 'score', 'is_new']]
        if c < break_count:
            break
    return df


if __name__ == "__main__":
    np.random.seed = 2
    data = np.random.randn(490, 10)
    res = main(data, iter_num=10)
    print res.shape
    print res[res['src'] == 0]
    print np.argsort(np.linalg.norm(data - data[0], axis=1))[:10]
