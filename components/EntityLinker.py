import pandas as pd

class EntityLinker:
    """
    EntityLinker(question), where question is the whole question string, would return a tuple(tokenized list of the question with entity replaced, entity)
    把q的topic entity用<e>替换
    !!! Remember that you have to escape the 's present in the string (use "..." or '''...''')
    """
    def __init__(self, path_KB, path_QA):
        try:
            self.df_qa = pd.read_csv(path_QA, sep='\t', header=None, names=['question_sentence', 'answer_set', 'answer_path'])
            self.df_qa['answer'] = self.df_qa['answer_set'].apply(lambda x: x.split('(')[0])
            self.df_qa['q_split'] = self.df_qa['question_sentence'].apply(lambda x: x.lower().split(' '))
            self.df_kb = pd.read_csv(path_KB,sep='\s', header=None, names= ['e_subject','relation','e_object'])
            self.create_entity_set()
        except Exception as e:
            print('File path wrong')
#entity_set是subject和object的并集，就是kb里面所有的entity嘛
    def create_entity_set(self):
        subject_set = set(self.df_kb['e_subject'].unique())
        object_set = set(self.df_kb['e_object'].unique())
        self.entity_set = subject_set.union(object_set)

    #Use on the question string and return a list of words from the question (replaced)
    def find_entity(self, question):
        modified_question_list = []
        entity_list = []
        entity = ''
        for idx, item in enumerate(question.split(' ')):
            if item in self.entity_set:
                entity_list.append(item)
        entity = max(entity_list, key=len)#entity_list中 len最大的entity
        '''
        不明白上面一句————将长度最大的entity作为topic entity，why????
        
        我猜测应该是，PQ里面的Q例子→  Q：what is the anna_of_holstein-gottorp 's children 's mother 's place of birth ?
        一般最长的那个带下划线的就是topic entitiy
        '''
        for item in question.split(' '):
            #将q中的topic entity用<e>替换掉
            if item == entity:
                modified_question_list.append('<e>')
            else:
                '''
                如果带下划线的一组词不是topic entity的话 就将其tokenized（分词）
                '''
                if len(item.split('_'))>0:
                    for x in item.split('_'):
                        if x != '':
                            modified_question_list.append(x)
                else:
                    modified_question_list.append(item)
        return (modified_question_list, entity)
