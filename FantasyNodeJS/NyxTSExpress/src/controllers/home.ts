import { Request, Response } from "express";

/**
 * GET / GET的路由中转
 * Home page.
 */
export const index = (req: Request, res: Response) => {
    res.render("home", {
        title: "首页"
    });
};
