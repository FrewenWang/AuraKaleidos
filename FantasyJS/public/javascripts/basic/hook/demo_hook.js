function student(name, score, praise) {
    return {
        name: name,
        score: score,
        praise: praise
    }
}

function praiseAdd(students) {
    const results = {};
    for (const i in students) {
        const curStudent = students[i];
        let ret = curStudent.score;
        if (curStudent.praise == 1) {
            ret += 20;
        } else if (curStudent.praise == 2) {
            ret += 10;
        } else if (curStudent.praise == 3) {
            ret += 5;
        }
        results[curStudent.name] = ret;
    }
    return results;
}


const liming = student("liming", 70, 1);
const liyi = student("liyi", 90, 2);
const liuwei = student("liuwei", 80, 3);
const ertuzi = student("ertuzi", 85, 3);

const result = praiseAdd([liming, liyi, liuwei, ertuzi]);
for (const i in result) {
    console.log("name:" + i + ",score:" + result[i]);
}


const praiseList = {
    1: 20,
    2: 10,
    3: 5
};


function praiseAdd(students) {
    const results = {};
    for (const i in students) {
        const curStudent = students[i];
        let ret = curStudent.score;
        if (praiseList[curStudent.praise]) {
            ret += praiseList[curStudent.praise];
        }
        results[curStudent.name] = ret;
    }
    return results;

}
