#include "definitionsHODLR.h"
#include "templatesHODLR.h"
#include "TestSuiteHODLR.h"

/*********************************
Implementation of Queue by List.
(We are going to use it later 
in PrintRanksInWidth function)
*********************************/

void init(struct my_queue* &q)
{
	q = (struct my_queue*)malloc(sizeof(struct my_queue));
	q->first = NULL;
	q->last = NULL;
}

void init(struct my_queue2* &q)
{
	q = (struct my_queue2*)malloc(sizeof(struct my_queue2));
	q->first = NULL;
	q->last = NULL;
}

bool my_empty(struct my_queue* q)
{
	if (q->first == NULL && q->last == NULL) return true;
	else return false;
}

bool my_empty(struct my_queue2* q)
{
	if (q->first == NULL && q->last == NULL) return true;
	else return false;
}

void push(struct my_queue* &q, cmnode* node)
{
	qlist *item = (qlist*)malloc(sizeof(qlist)); // просто перекидываем указатель по листу, память под node уже выделена на верхнем уровне
	item->node = node;
	item->next = NULL;

	// указатель first установлен всегда на первый элемент, а last - двигается по списку по мере добавления элементов, но всегда
	// указывает на последний элемент

	// Если очередь пуста
	if (my_empty(q))
	{ 
		q->first = item;
		q->last = q->first;
	}
	else
	{
#if 0
		if (q->first == q->last)
		{
			q->first->next = item;
			q->last = item;
		}
		else
		{
			q->last->next = item;
			q->last = item;
		}
#else
		q->last->next = item; // кладем в указатель последнего элемента новый item
		q->last = item; // т.к. добавился в конец новый элемент, то переносим указатель last на него
#endif
	}
}

void push(struct my_queue2* &q, cumnode* node)
{
	qlist2 *item = (qlist2*)malloc(sizeof(qlist2)); // просто перекидываем указатель по листу, память под node уже выделена на верхнем уровне
	item->node = node;
	item->next = NULL;

	// указатель first установлен всегда на первый элемент, а last - двигается по списку по мере добавления элементов, но всегда
	// указывает на последний элемент

	// Если очередь пуста
	if (my_empty(q))
	{
		q->first = item;
		q->last = q->first;
	}
	else
	{
		q->last->next = item; // кладем в указатель последнего элемента новый item
		q->last = item; // т.к. добавился в конец новый элемент, то переносим указатель last на него
	}
}

// удаляем первый элемент из очереди
void pop(struct my_queue* &q)
{
	qlist* temp; // память уже выделена в insert под node, просто перекидываем указатель
	if (my_empty(q)) return;
	else
	{
		temp = q->first; // сохраняем старый первый номер
		q->first = q->first->next;
		if (q->first == NULL) q->last = NULL; // для удаления последнего элемента

		free(temp); // удаляем только указатель, память под mnode удаляется на другом уровне
	}
}

void pop(struct my_queue2* &q)
{
	qlist2* temp; // память уже выделена в insert под node, просто перекидываем указатель
	if (my_empty(q)) return;
	else
	{
		temp = q->first; // сохраняем старый первый номер
		q->first = q->first->next;
		if (q->first == NULL) q->last = NULL; // для удаления последнего элемента

		free(temp); // удаляем только указатель, память под mnode удаляется на другом уровне
	}
}

cmnode* front(struct my_queue* q)
{
	cmnode* x;
	x = q->first->node;
	return x;
}

cumnode* front(struct my_queue2* q)
{
	cumnode* x;
	x = q->first->node;
	return x;
}

void print_queue(struct my_queue* q)
{
	qlist* h;
	if (my_empty(q))
	{
		printf("Queue is empty!\n");
		return;
	}
	else
	{
		printf("Cur queue: ");
		for (h = q->first; h != NULL; h = h->next)
			printf("%d ", h->node->p);
		printf("\n");
	}
}
