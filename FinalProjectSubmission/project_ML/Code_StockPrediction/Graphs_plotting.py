import matplotlib.pyplot as plt
import numpy as np


def line_graph(fig_no, test_Y, predictions,model,stock):
    # Plots a line graph given the x and y axes
    values = test_Y
    labels =["Actuals", "Predictions"]

    # Plot a line graph
    plt.figure(fig_no, figsize=(6, 4))  # 6x4 is the aspect ratio for the plot
    plt.plot(np.arange(0, len(test_Y)), values, 'ob-', linewidth=3)  # Plot the first series in red with circle marker
    plt.plot(np.arange(0, len(test_Y)), predictions, 'or-', linewidth=3)  # Plot the second series in blue with square marker

    # This plots the data
    plt.grid(True)  # Turn the grid on
    plt.ylabel("Closing Price")  # Y-axis label
    plt.xlabel("Day Count")  # X-axis label
    #plt.title("Test Actuals and Predicted Values")  # Plot title
    plt.title(model + stock)
    plt.legend(labels, loc="best")
    # Make sure labels and titles are inside plot area
    plt.tight_layout()
    # Save the chart
    plt.savefig("../Figures/line_plot_" + "{}".format(stock) + "_" + "{}".format(model) +".pdf")
    #plt.show()


def bar_chart(fig_no, rmse, model_names,  stock, model_count):
     #Plots the bar graph given the lables, values and titles. saves it in the given filename.
    values = rmse
    inds = np.arange(model_count)
    labels = model_names
    # Plot a bar chart
    plt.figure(fig_no, figsize=(20, 10))  # 6x4 is the aspect ratio for the plot
    plt.bar(inds, values, align='center')  # This plots the data
    plt.grid(True)  # Turn the grid on
    plt.ylabel("RMSE")  # Y-axis label
    plt.xlabel("Regressor Models")  # X-axis label
    plt.title("{}".format(stock) + " RMSE")  # Plot title
    # plt.xlim(xliml, xlimu)  # set x axis range
    # plt.ylim(yliml, ylimu)  # Set yaxis range
    # Set the bar labels
    plt.gca().set_xticks(inds)  # label locations
    plt.gca().set_xticklabels(labels)  # label values
    # Make sure labels and titles are inside plot area
    plt.tight_layout()
    # Save the chart
    plt.savefig("../Figures/{}.pdf".format(stock))
    #plt.show()

