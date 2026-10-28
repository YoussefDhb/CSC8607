# Compte Rendu — TP4 : Réseaux récurrents et génération de séquences
**Cours :** CSC 8607 — Introduction au Deep Learning  
**Auteur :** Youssef DHOUIB

---

## Exercice 1 : Exploration et Prétraitement des partitions ABC

### Output du script train_music_ex1.py

![Exercice 1](ex1.png)

### Étape 1 : Chargement et découverte du jeu de données

### Question 1.1

- **Exécutez ce code et indiquez dans votre rapport le nombre exact de partitions présentes dans `train.json` et dans `validation.json`.**

    *   `train.json` : **[À COMPLÉTER]** partitions
    *   `validation.json` : **[À COMPLÉTER]** partitions

### Question 1.2

- **Copiez le texte de la première partition (`train_data[0]["abc notation"]`) et collez-le dans le lecteur interactif ABC Player pour voir la partition et l'écouter. En observant le texte brut, identifiez et expliquez le rôle des métadonnées d'en-tête présentes sur les premières lignes avant le début des notes.**

    ```text
    [À COMPLÉTER : texte de train_data[0]["abc notation"]]
    ```

    ![Partition ABC Player](ex1_abc.png)

    Les premières lignes, de la forme `Lettre:valeur`, sont l'**en-tête** de la partition : elles ne contiennent pas de notes, mais donnent au lecteur les informations nécessaires pour interpréter correctement les notes qui suivent.
    *   `X:` : **numéro de référence** du morceau dans le fichier (toujours la première ligne).
    *   `T:` : **titre** du morceau.
    *   `M:` : **métrique** (ex. `6/8`, `4/4`), c'est-à-dire le nombre et la valeur des temps dans une mesure ; elle fixe la longueur de chaque mesure entre deux `|`.
    *   `L:` : **durée de note par défaut** (ex. `1/8` = croche) : une note écrite sans chiffre a cette durée, `A2` dure deux fois plus, `A/2` deux fois moins.
    *   `K:` : **tonalité** (ex. `G`, `Dmaj`), qui fixe les altérations à appliquer automatiquement aux notes ; c'est toujours la dernière ligne d'en-tête, les notes commencent juste après.
    *   [À COMPLÉTER selon la partition : autres champs présents, par ex. `R:` (rythme : reel, jig…), `C:` (compositeur), `Q:` (tempo)]

    -> Sans cet en-tête, la même suite de lettres ne peut pas être jouée : le lecteur ne connaîtrait ni la durée des notes, ni la mesure, ni la tonalité.

### Étape 2 : Construction du vocabulaire et Padding

### Question 1.3

- **Combien y a-t-il de caractères uniques (`vocab_size`) ? Quelle est la longueur moyenne et la longueur maximale d'une partition dans le jeu d'entraînement ?**

    *   `vocab_size` = **[À COMPLÉTER]** caractères uniques
    *   Longueur moyenne : **[À COMPLÉTER]** caractères
    *   Longueur maximale : **[À COMPLÉTER]** caractères

### Question 1.4

- **Pourquoi est-il indispensable d'appliquer `sorted(...)` sur le `set` des caractères uniques avant de créer `char_to_index` ? Que se passerait-il si vous sauvegardiez les poids de votre modèle aujourd'hui et que vous relanciez votre script demain pour générer de la musique sans ce `sorted()` ?**

    Un `set` n'a **pas d'ordre garanti** : en Python, le hachage des chaînes de caractères est **aléatoire à chaque lancement** de l'interpréteur (`PYTHONHASHSEED`), donc `list(set(all_text))` peut renvoyer les caractères dans un ordre différent d'une exécution à l'autre. `sorted()` rend l'ordre **déterministe** : le même caractère reçoit toujours le même indice.

    Sans `sorted()`, le modèle sauvegardé aujourd'hui aurait appris avec une correspondance (par ex. `'A'` → 12) et le script relancé demain en construirait une autre (`'A'` → 47). Les lignes de la matrice d'`Embedding` et les sorties de la couche `fc` ne correspondraient plus aux bons caractères : le modèle lirait une amorce « mélangée » et ses prédictions seraient décodées vers les mauvais caractères, ce qui produirait une suite de caractères sans aucun sens.

    -> Le vocabulaire fait partie du modèle au même titre que ses poids : il doit être reconstruit exactement à l'identique (ou sauvegardé avec les poids).

### Question 1.5

- **Pourquoi avons-nous choisi de tronquer à `SEQ_LENGTH = 500` (environ deux fois la longueur moyenne) plutôt que d'aligner toutes les chansons sur la longueur maximale `max_len` du dataset ? Quel serait l'impact d'un padding jusqu'à `max_len` sur le temps de calcul, la mémoire GPU et l'apprentissage du modèle ?**

    Avec `SEQ_LENGTH = 500`, la **grande majorité des partitions** tient entièrement dans la séquence, et seules les plus longues perdent leur fin. Aligner tout le monde sur `max_len` = **[À COMPLÉTER]**, soit environ **[À COMPLÉTER]** fois plus, ferait payer à toutes les partitions le coût de la plus longue :
    *   **Temps de calcul :** un RNN traite la séquence pas à pas, de façon séquentielle : le temps du *forward* et de la BPTT est proportionnel à la longueur, donc environ **[À COMPLÉTER]** fois plus long par batch, essentiellement pour traiter des espaces.
    *   **Mémoire GPU :** pour la BPTT, il faut stocker les activations de chaque pas de temps (taille $\text{batch} \times L \times \text{hidden\_size}$), donc la mémoire croît aussi linéairement avec $L$ et obligerait à réduire fortement le batch.
    *   **Apprentissage :** la plupart des positions seraient du padding. La loss serait dominée par la prédiction triviale « espace → espace », ce qui dilue le signal utile des vraies notes, et des séquences très longues aggravent le *vanishing* / *exploding gradient* pendant la BPTT.

    -> 500 caractères est un compromis : on garde presque toute l'information musicale pour un coût de calcul et de mémoire raisonnable.