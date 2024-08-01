import sqlite3
import numpy as np
from src.utils import *
from typing import Optional


class DatabaseOperations:
    """
    A class used to perfome operation on database

    ...

    Attributes
    ----------
    database_connection : sqlite3.Connection
        a connection to the sqlite database
    datablock_reference : str
        the table name of the sqllite database

    """

    def __init__(
        self,
        database_connection: sqlite3.Connection,
        datablock_reference: str,
        vector_datatype: Optional[str] = None,
        vector_dimension: Optional[int] = None,
    ) -> None:
        self.database_connection = database_connection
        self.datablock_reference = datablock_reference
        self.vector_datatype = vector_datatype
        self.vector_dimension = vector_dimension

    def _store_datablock_and_vector_details(self) -> None:
        if self.vector_datatype == None or self.vector_dimension == None:
            raise ValueError("please specifiy vector datatype and dimension explicitly")
        else:
            script: str = (
                f"INSERT INTO dbDetails (datablockReference,vectorDatatype,vectorDimension) VALUES (?,?,?)"
            )

            self.database_connection.execute(
                script,
                (self.datablock_reference, self.vector_datatype, self.vector_dimension),
            )

    def _get_datablock_and_vector_details(self) -> None:
        script: str = (
            f"SELECT vectorDatatype, vectorDimension FROM dbDetails WHERE datablockReference = ?"
        )
        generator = self.database_connection.execute(
            script, (self.datablock_reference,)
        )
        try:
            result = next(generator)
        except Exception as e:
            print(f"Error: {e}")
            raise Exception("database got un-expected error")

        self.vector_datatype, self.vector_dimension = result[0], result[1]

    def add(self, text: str, vector: np.ndarray) -> None:
        """
        A method used to add a single entity to database

        ...

        Parameters
        ----------
        text : str
        vector : np.ndarray

        """
        if not isinstance(text, str) or not isinstance(vector, np.ndarray):
            raise ValueError("un-expected type provided")
        if self.vector_datatype == None or self.vector_dimension == None:
            self.vector_datatype = str(vector.dtype)
            self.vector_dimension = vector.shape[-1]
        self._store_datablock_and_vector_details()

        id: str = get_id()
        data_bytes = convert_numpy_array_to_bytes(vector)
        script = (
            f"INSERT INTO {self.datablock_reference} (id,text,vector) VALUES (?,?,?)"
        )
        try:
            self.database_connection.execute(script, (id, text, data_bytes))

            self.database_connection.commit()
        except Exception as e:
            self.database_connection.rollback()
            print(f"Error occurred: {e}")
            raise RuntimeError("Not able to add the entity .")

    def add_many(self, texts: list[str], vectors: np.ndarray) -> None:
        """
        A method used to add multiple entities to database

        ...

        Parameters
        ----------
        texts : list[str]
            A list of texts associated with the vectors.
        vectors : np.ndarray
            A numpy array of vectors to be stored in the database.


        """
        if not isinstance(texts, list) or not isinstance(vectors, np.ndarray):
            raise ValueError("un-expected type provided")
        if self.vector_datatype == None or self.vector_dimension == None:
            self.vector_datatype = str(vectors.dtype)
            self.vector_dimension = vectors.shape[-1]
        self._store_datablock_and_vector_details()

        def prepare_data(tup: tuple) -> tuple:
            index, vector = tup

            id: str = get_id()
            data_bytes = convert_numpy_array_to_bytes(vector)
            return (id, texts[index], data_bytes)

        data_for_store_operation: list[tuple] = list(
            map(prepare_data, enumerate(vectors))
        )

        script = (
            f"INSERT INTO {self.datablock_reference} (id,text,vector) VALUES (?,?,?)"
        )
        try:
            self.database_connection.executemany(script, data_for_store_operation)
            self.database_connection.commit()
        except Exception as e:
            self.database_connection.rollback()
            print(f"Error occurred: {e}")
            raise RuntimeError("Not able to add the entities .")

    def update(self, id: str, text: str, vector: np.ndarray) -> None:
        """
        A method used to update a single entity in database

        ...

        Parameters
        ----------
        id : str
            the id of the entity that is going to update
        text : str
            the updated text
        vector : np.ndarray
            the updated vector

        """
        if (
            not isinstance(id, str)
            or not isinstance(text, str)
            or not isinstance(vector, np.ndarray)
        ):
            raise ValueError("un-expected type provided")

        data_bytes: bytes = convert_numpy_array_to_bytes(vector)
        script: str = (
            f"UPDATE {self.datablock_reference} SET text = ?, vector = ? WHERE id = ?"
        )
        try:
            self.database_connection.execute(script, (text, data_bytes, id))

            self.database_connection.commit()
        except Exception as e:
            self.database_connection.rollback()
            print(f"Error occurred: {e}")
            raise RuntimeError("Not able to update the entity .")

    def update_many(
        self, ids: list[str], texts: list[str], vectors: np.ndarray
    ) -> None:
        """
        A method used to update a multiple entities in database

        ...

        Parameters
        ----------
        ids : list[str]
            the list of id's of entities that are going to update
        texts : list[str]
            the list of updated text
        vectors : np.ndarray
            the array of updated vectors

        """
        if (
            not isinstance(ids, list)
            or not isinstance(texts, list)
            or not isinstance(vectors, np.ndarray)
        ):
            raise ValueError("un-expected type provided")
        if len(ids) != len(texts) or len(ids) != vectors.shape[0]:
            raise ValueError("invalid inputs are passed.")

        def prepare_data(tup: tuple) -> tuple:
            id, text, vector = tup
            data_bytes: bytes = convert_numpy_array_to_bytes(vector)
            return (text, data_bytes, id)

        data_for_update_operation: list[tuple] = list(
            map(prepare_data, zip(ids, texts, vectors))
        )

        script: str = (
            f"UPDATE {self.datablock_reference} SET text = ?, vector = ? WHERE id = ?"
        )
        try:
            self.database_connection.executemany(script, data_for_update_operation)

            self.database_connection.commit()
        except Exception as e:
            self.database_connection.rollback()
            print(f"Error occurred: {e}")
            raise RuntimeError("Not able to update the entities .")

    def delete(self, id: str) -> None:
        """
        A method used to delete a single entity in database

        ...

        Parameters
        ----------
        id : str
            the id of the entity that is going to delete.

        """
        if not isinstance(id, str):
            raise ValueError("un-expected type provided")
        script: str = f"DELETE FROM {self.datablock_reference} WHERE id = ?"
        try:
            self.database_connection.execute(script, (id,))

            self.database_connection.commit()
        except Exception as e:
            self.database_connection.rollback()
            print(f"Error occurred: {e}")
            raise RuntimeError("Not able to delete the entity .")

    def delete_many(self, ids: list[str]) -> None:
        """
        A method used to delete a multiple entities in database

        ...

        Parameters
        ----------
        id : list[str]
            the list of id's of entities that are going to delete

        """
        if not isinstance(ids, list):
            raise ValueError("un-expected type provided")

        def modify_id(id: str) -> tuple:
            return (id,)

        ids = list(map(modify_id, ids))
        script: str = f"DELETE FROM {self.datablock_reference} WHERE id = ?"
        try:
            self.database_connection.executemany(script, ids)

            self.database_connection.commit()
        except Exception as e:
            self.database_connection.rollback()
            print(f"Error occurred: {e}")
            raise RuntimeError("Not able to delete the entities .")

    def delete_all(
        self,
    ) -> None:
        """
        A method used to delete all entities in database


        """

        script: str = f"DELETE FROM {self.datablock_reference}"
        try:
            self.database_connection.execute(script)

            self.database_connection.commit()
        except Exception as e:
            self.database_connection.rollback()
            print(f"Error occurred: {e}")
            raise RuntimeError("Not able to delete all the entities .")

    def get_by_id(self, id: str) -> tuple:
        """
        A method used to get specific entity in database

        ...

        Parameters
        ----------
        id : str
            the id of the entity that we want..

        """
        if not isinstance(id, str):
            raise ValueError("un-expected type provided")
        self._get_datablock_and_vector_details()

        script: str = f"SELECT * FROM {self.datablock_reference} WHERE id = ?"
        try:
            generator = self.database_connection.execute(script, (id,))
        except Exception as e:
            print(f"Error occurred: {e}")
            raise RuntimeError(
                f"internal error.not able to get the entity by id: {id}."
            )
        try:
            entity: tuple = next(generator)
        except Exception as e:
            print(f"Error: {e}")
            raise RuntimeError("given id doesn't match with any id in db")
        entity: tuple = (
            entity[0],
            entity[1],
            convert_bytes_to_numpy_array(
                entity[2], self.vector_datatype, self.vector_dimension
            ),
        )
        return entity

    def search(
        self, vector: np.ndarray, vectorsearch_algo: callable, top_k: int
    ) -> list[tuple]:
        """
        A method used to search similar top_k entities in database

        ...

        Parameters
        ----------
        vector : np.ndarray
            the vector for which top_k search is happening.
        vectorsearch_algo : callable
            the algorithm on which search will happen.
        top_k : int
            the total number of search results

        """
        if (
            not isinstance(vector, np.ndarray)
            or not callable(vectorsearch_algo)
            or not isinstance(top_k, int)
        ):
            raise ValueError("un-expected type provided")
        self._get_datablock_and_vector_details()

        script: str = f"SELECT * FROM {self.datablock_reference}"
        try:
            stored_entities = self.database_connection.execute(script)
        except Exception as e:
            print(f"Error occurred: {e}")
            raise RuntimeError("internal error.Not able to perfome serach.")

        unranked_topk_data_arr: list[tuple] = [((), 1000.0)] * top_k

        while True:
            try:
                stored_entity = next(stored_entities)
                stored_id = stored_entity[0]
                stored_text = stored_entity[1]
                stored_vector = convert_bytes_to_numpy_array(
                    stored_entity[-1], self.vector_datatype, self.vector_dimension
                )

                distance: float = vectorsearch_algo(vector, stored_vector)

                flag_value_for_update: bool = True
                highest_distance_available: float = max(
                    [i[1] for i in unranked_topk_data_arr]
                )

                def update_if_smaller(tuple_ele: tuple) -> tuple:
                    nonlocal flag_value_for_update
                    collected_distance = tuple_ele[1]
                    if collected_distance == highest_distance_available:
                        if (distance < collected_distance) and (
                            flag_value_for_update == True
                        ):
                            flag_value_for_update = False
                            return ((stored_id, stored_text, stored_vector), distance)
                    return tuple_ele

                unranked_topk_data_arr: list[tuple] = list(
                    map(update_if_smaller, unranked_topk_data_arr)
                )
            except Exception as e:
                print(e)
                break
        ranked_topk_data_arr: list[tuple] = sorted(
            unranked_topk_data_arr, key=lambda x: x[1]
        )
        return ranked_topk_data_arr


class Utilities:
    @staticmethod
    def convert_float_to_binary(float_vec: np.ndarray) -> np.ndarray:
        """
        A method used to convert float vector to binaru vector
        ...

        Parameters
        ----------
        float_vec : np.ndarray
            the float vector

        """
        if not isinstance(float_vec, np.ndarray):
            raise ValueError("un-expected type provided")

        binary_vec = np.zeros_like(float_vec)
        binary_vec[float_vec > 0] = 1
        binary_vec[float_vec <= 0] = 0
        return binary_vec.astype(np.uint8)


class VectorSearchAlgorithms:
    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        A method used to find (1-cosine similarity) between two vector
            distance = 1 - cos(theta)
        ...

        Parameters
        ----------
        vec1 : np.ndarray
        vec2 : np.ndarray

        """
        if not isinstance(vec1, np.ndarray) or not isinstance(vec2, np.ndarray):
            raise ValueError("un-expected type provided")

        vec1, vec2 = vec1.squeeze(), vec2.squeeze()
        if vec1.ndim != 1 or vec2.ndim != 1:
            raise ValueError(
                "vectors are not compatible with these vector serach algorithm"
            )
        result: float = round(
            float(
                1 - (np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
            ),
            3,
        )

        return result

    def normalized_hamming_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        A method used to find (hamming distance) between two vector
            distance = summation(1,{where vec1[i] != vec2[i]})
        ...

        Parameters
        ----------
        vec1 : np.ndarray
        vec2 : np.ndarray

        """
        if not isinstance(vec1, np.ndarray) or not isinstance(vec2, np.ndarray):
            raise ValueError("un-expected type provided")

        vec1, vec2 = vec1.squeeze(), vec2.squeeze()
        if vec1.ndim != 1 or vec2.ndim != 1 or vec1.shape[0] != vec2.shape[0]:
            raise ValueError(
                "vectors are not compatible with these vector serach algorithm"
            )
        vector_length: int = vec1.shape[0]
        bool_vec: np.ndarray = vec1 != vec2
        result: float = round(int(np.count_nonzero(bool_vec)) / vector_length, 3)
        return result


def start_connection(database_name: str):
    """
    A method used to start the connection
    ...

    Parameters
    ----------
    database_name : str
        The name for the new vector database.

    """
    conn: sqlite3.Connection = sqlite3.connect(database_name)

    script: str = (
        f"CREATE TABLE IF NOT EXISTS dbDetails (datablockReference TEXT,vectorDatatype TEXT,vectorDimension INTEGER)"
    )
    conn.execute(script)
    conn.commit()
    return conn


def create_datablock(datablock_name: str, database_connection: sqlite3.Connection):
    """
    A method used to create the datablock
    ...

    Parameters
    ----------
    datablock_name : str
        The name for the new datablock.
    database_connection : sqlite3.Connection
        The connection to the database

    """
    if datablock_name == "dbDetails":
        raise ValueError(
            "dbDetails is a reserved datablock.please go for another datablock name"
        )
    script: str = (
        f"CREATE TABLE IF NOT EXISTS {datablock_name} (id TEXT,text TEXT,vector BLOBS)"
    )
    database_connection.execute(script)
    database_connection.commit()


def close_connection(database_connection: sqlite3.Connection):
    """
    A method used to close the connection
    ...

    Parameters
    ----------
    database_connection : sqlite3.Connection
        The connection to your database.

    """
    database_connection.close()
