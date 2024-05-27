def plot_hist_pls_scores(pls_pipe, pc_x_axis=1, pc_y_axis=2, X_columns=None):
    model = pls_pipe["model"]
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"colspan": 2}, None], [{}, {}]],
        subplot_titles=(
            "Scores Plot ",
            "Loadings of Principal Component - " + str(pc_x_axis),
            "Loadings of Principal Component - " + str(pc_y_axis),
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=model.x_scores_[:, pc_x_axis],
            y=model.x_scores_[:, pc_y_axis],
            mode="markers",
            name="Scores",
        ),
        row=1,
        col=1,
    )
    fig.add_bar(
        x=X_columns,
        y=model.x_loadings_[:, pc_x_axis - 1],
        name="Loadings PC - " + str(pc_x_axis),
        row=2,
        col=[1, 2],
    )
    fig.add_bar(
        x=X_columns,
        y=model.x_loadings_[:, pc_y_axis - 1],
        name="Loadings PC - " + str(pc_y_axis),
        row=2,
        col=2,
    )
    fig.update_layout(height=1000)
    fig.show()
