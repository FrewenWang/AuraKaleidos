import request from "supertest";
import app from "../src/app";
import { expect } from "chai";
import { User } from "../src/models/User";

describe("GET /login", () => {
    it("should return 200 OK", () => {
        return request(app).get("/login")
            .expect(200);
    });
});


describe("GET /forgot", () => {
    it("should return 200 OK", () => {
        return request(app).get("/forgot")
            .expect(200);
    });
});

describe("GET /signup", () => {
    it("should return 200 OK", () => {
        return request(app).get("/signup")
            .expect(200);
    });
});

describe("GET /reset", () => {
    it("should return 302 Found for redirection", async () => {
        const query: any = {
            where: () => query,
            gt: () => query,
            exec: (callback: Function) => callback(undefined, null)
        };
        const findOne = jest.spyOn(User, "findOne").mockReturnValue(query);
        try {
            await request(app).get("/reset/1").expect(302);
        } finally {
            findOne.mockRestore();
        }
    });
});

describe("POST /login", () => {
    it("should return some defined error message with valid parameters", (done) => {
        return request(app).post("/login")
            .field("email", "john@me.com")
            .field("password", "Hunter2")
            .expect(302)
            .end(function(err, res) {
                expect(res.error).not.to.be.undefined;
                done();
            });

    });
});
