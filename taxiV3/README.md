# T-AIA-902
Projet : Apprentissage par renforcement avec Taxi-v3

Ce projet implémente un agent basé sur l’algorithme Q-learning pour résoudre l’environnement Taxi-v3 de la librairie Gymnasium (anciennement Gym).

    L’objectif de l’environnement Taxi-v3 est de déplacer un taxi (représenté par une lettre “yellow” dans la grille) pour prendre un passager et le déposer au bon endroit, tout en optimisant le nombre de mouvements.

Sommaire

    Fonctionnalités
    Prérequis
    Installation
    Comment exécuter
    Détails de l’algorithme
    Visualiser les résultats
    Structure du projet
    Aller plus loin
    Licence

Fonctionnalités

    Entraînement d’un agent avec Q-learning sur Taxi-v3.
    Sauvegarde et chargement de la table de Q-valeurs dans un fichier taxi.pkl (via la librairie pickle).
    Visualisation de la performance de l’agent au fil des épisodes (courbe de somme des récompenses).
    Exécution en mode test avec affichage du rendu (render) pour observer l’agent en action.

Prérequis

    Python 3.7 ou version supérieure
    Gymnasium (pour l’environnement Taxi-v3)
    NumPy
    Matplotlib
    pickle (inclus nativement dans Python)

Installation

    Cloner ce dépôt ou téléchargez les fichiers dans un dossier local.
    Ouvrez un terminal dans ce dossier et installez les dépendances :

    pip install gymnasium numpy matplotlib

    Si vous utilisez un environnement virtuel (recommandé), activez-le au préalable.

Comment exécuter

Le script principal s’appelle par exemple taxi.py (ou autre nom équivalent). Il contient deux étapes :

    Entraînement de l’agent pendant 15 000 épisodes (avec génération de la Q-table).
    Test de l’agent pendant 10 épisodes avec affichage du rendu graphique.

Exécutez :

python taxi.py

Par défaut, la fonction run() est appelée deux fois :

    En mode entraînement : run(15000)
    En mode test : run(10, is_training=False, render=True)

Détails de l’algorithme

Le code implémente l’algorithme Q-learning, décrit par les équations suivantes :
Q(s,a)←Q(s,a)  +  α(r+γmax⁡a′Q(s′,a′)  −  Q(s,a))
Q(s,a)←Q(s,a)+α(r+γa′max​Q(s′,a′)−Q(s,a))

    ss : l’état actuel (position du taxi, du passager, etc.).
    aa : l’action entreprise (0=aller à gauche, 1=descendre, 2=aller à droite, 3=monter, etc.).
    rr : la récompense reçue après avoir effectué l’action aa.
    s′s′ : le nouvel état après l’action.
    αα (learning rate) : taux d’apprentissage (définit à 0.9 au départ).
    γγ (discount factor) : facteur de pondération des récompenses futures (définit à 0.9).
    ϵϵ : facteur d’exploration (au départ 1, puis décroît progressivement).

Stratégie ϵϵ-greedy

    Avec une probabilité ϵϵ, on explore : on choisit une action aléatoire.
    Avec une probabilité 1−ϵ1−ϵ, on exploite : on choisit l’action qui maximise la Q-valeur courante.
    ϵϵ décroît au fil des épisodes, afin de privilégier la découverte au début et l’exploitation à la fin.

Visualiser les résultats

    Graphique : À la fin de l’exécution en mode entraînement, un fichier taxi.png est créé. Il représente la somme des récompenses sur les 100 derniers épisodes pour chaque épisode.
    Rendu graphique : Lors du mode test (avec render=True), une fenêtre s’ouvre (ou un affichage texte dans le terminal selon la version de Gymnasium) pour montrer l’évolution du taxi dans la grille.

Structure du projet

.
├── taxi.py             # Script principal, contenant l'apprentissage et le test
├── taxi.pkl            # Fichier contenant la Q-table (généré après l'entraînement)
├── taxi.png            # Graphique des performances (généré après l'entraînement)
└── README.md           # Ce fichier de documentation

Aller plus loin

    Hyperparamètres : Vous pouvez ajuster learning_rate_a, discount_factor_g, epsilon, etc. pour étudier leur impact sur la vitesse et la qualité d’apprentissage.
    Politique d’exploration : Remplacer la stratégie ϵϵ-greedy par d’autres méthodes (Boltzmann, etc.).
    Environnements : Tester l’algorithme sur d’autres environnements Gymnasium pour comparer les performances de Q-learning.
    Algorithmes : Expérimenter avec d’autres approches d’apprentissage par renforcement (SARSA, Deep Q-Network, etc.).

Licence

Ce projet est disponible sous licence libre (à adapter selon vos besoins ou licence choisie).
Vous êtes libre de le cloner, le modifier et de le partager.

Bonne exploration et bon apprentissage avec Taxi-v3 !