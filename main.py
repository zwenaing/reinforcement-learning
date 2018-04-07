import sys, getopt
from q_learning import QLearning


if __name__ == '__main__':
    arguments = sys.argv[1:]
    try:
        opts, args = getopt.getopt(arguments, "h:alg:size:gamma:exps:epsilon:eps:alpha:lambda")
    except getopt.GetoptError:
        print("main.py -alg <q_learning_startq or s_learning_startq> -size <size-of-grid> -gamma <discount-factor> -exps <number-of-experiments>"
              "-eps <number-of-epsisodes> -epsilon <epsilon-in-greedy-policy-execution> -alpha <learning-rate> "
              "-lambda <parameter-for-Sarsa>")

    alg = 'q_learning_startq'
    size = 5
    gamma = 0.99
    exps = 500
    eps = 500
    epsilon = 0.1
    alpha = 0.1
    sarsa_lambda = 0

    for opt, arg in opts:
        if opt == "-h":
            print("main.py -alg <q_learning_startq or s_learning_startq> -size <size-of-grid> -gamma <discount-factor> -exps <number-of-experiments>"
                  "-eps <number-of-epsisodes> -epsilon <epsilon-in-greedy-policy-execution> -alpha <learning-rate> "
                  "-lambda <parameter-for-Sarsa>")
        if opt == "-alg":
            alg = arg
        elif opt == "-size":
            size = arg
        elif opt == "-gamma":
            gamma = arg
        elif opt == "-exps":
            exps = arg
        elif opt == "-epsilon":
            epsilon = arg
        elif opt == "-eps":
            eps = arg
        elif opt == "-alpha":
            alpha = arg
        elif opt == "-lambda":
            sarsa_lambda = arg

    if alg == "q_learning_startq":
        algorithm = QLearning(size, gamma, exps, eps, epsilon, alpha)
        algorithm.fit()
    # print(alg + " " + str(size) + " " + str(gamma) + " " + str(exps) + " " + str(eps) + " " +
    #       str(epsilon) + " " + str(alpha) + " " + str(sarsa_lambda))